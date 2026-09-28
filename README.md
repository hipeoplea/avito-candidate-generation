# Avito Candidate Generation

Решение тестового задания по генерации кандидатов для поиска в категории услуг.

Для каждого поискового запроса требуется выбрать до 50 объявлений из общего корпуса.  
Целевая метрика — `Recall@50`.

## Структура проекта

```text
.
├── candidate generation.ipynb      # основной pipeline и генерация answer.csv
├── train_e5.ipynb                  # двухэтапный fine-tuning multilingual E5
├── answer.csv
├── e5_avito_pilot2_final/          # финальный checkpoint E5
├── artifacts/                      # retrieval-кэши и CatBoost model
└── dataset/                        # локально, не хранится в Git
```

## Общая схема решения

Pipeline состоит из двух частей:

1. генерация широкого набора кандидатов несколькими retrieval-моделями;
2. reranking top-300 кандидатов с помощью `CatBoostRanker`.

Используются четыре retrieval-канала:

- fine-tuned `intfloat/multilingual-e5-small`;
- word-level TF-IDF;
- character-level TF-IDF;
- TF-IDF по `title + params + description`.

Каждый канал возвращает top-1000. После объединения кандидаты предварительно ранжируются по semantic similarity, location prior и слабому lexical signal. Затем CatBoost выбирает финальные top-50.

---

## Как формировалась архитектура

### Анализ данных

Только `907 / 2452 = 36.99%` benchmark-запросов уже встречались в train по `search_query`, а полное совпадение всех query-признаков есть только для 108 запросов. Кроме того, лишь `6.63%` train-строк содержат item из benchmark corpus. Поэтому решение нельзя строить на запоминании query-item пар.

Категория также оказалась слабым фильтром: хотя в train `search_category == item_category_id` для 99.99% positive-пар, в benchmark почти весь corpus относится к категории `114`.

Location, наоборот, оказался сильным сигналом: exact match наблюдается в 83.10% positive-пар. Но жёстко фильтровать по location нельзя, поэтому используется soft prior:

```text
P(item_location_id | search_location_id)
```

Weighted coverage наиболее частых item locations:

| Top-K locations | Coverage |
|---:|---:|
| 1 | 90.31% |
| 5 | 95.61% |
| 10 | 97.16% |

Microcategory prior тоже проверялся, но после fine-tuning E5 почти не улучшал Recall@50, поэтому в финальную схему не вошёл.

---

## Валидация

Validation повторяет структуру benchmark:

```text
seen:    907
unseen: 1545
```

Для `seen` скрытая query-item пара удаляется, но query text остаётся в train.  
Для `unseen` query text полностью исключается из train.

Это позволяет избежать завышенной оценки из-за повторяющихся запросов и отдельно проверить перенос на новые формулировки.

---

## Dense retrieval и fine-tuning E5

В качестве dense retriever выбран `intfloat/multilingual-e5-small`: модель мультиязычная, подходит для retrieval и достаточно компактна для локального запуска на MPS.

Используемые тексты:

```text
query: <search_query>. <search_infm_params_text>
passage: <item_title_raw>. <item_infm_params_text>
```

Zero-shot E5:

| K | Recall |
|---:|---:|
| 50 | 0.3120 |
| 1000 | 0.7529 |

Высокая разница между Recall@50 и Recall@1000 показала, что важны и широкий candidate pool, и отдельный reranking.

### FT1

Fine-tuning выполняется через `MultipleNegativesRankingLoss` и `BatchSamplers.NO_DUPLICATES`.

```text
pairs = 104 063
epochs = 1
batch_size = 32
learning_rate = 2e-5
max_seq_length = 128
```

Параметры выбирались как практический компромисс между качеством и временем обучения на MPS.

После FT1:

```text
Recall@50   = 0.3601
Recall@1000 = 0.8544
```

### FT2

Для query с несколькими positive items на втором этапе по возможности используется другой positive. Альтернативный positive удалось подобрать для `36 112 / 104 063 = 34.70%` запросов.

FT2 продолжает обучение FT1 с меньшим `learning_rate=1e-5`.

После FT2:

```text
Recall@50   = 0.3679
Recall@1000 = 0.8613
```

Поэтому в финальном pipeline используется `e5_avito_pilot2_final`.

---

## Почему добавлены TF-IDF retrieval-каналы

Dense retrieval хорошо ловит смысл, но lexical методы лучше работают с точными названиями услуг, редкими словами, числами и опечатками.

### Word TF-IDF

```text
ngram_range = (1, 2)
min_df = 2
max_features = 200_000
sublinear_tf = True
```

Standalone `Recall@1000 = 0.6060`, но в union:

```text
FT2 dense:      0.8613
FT2 ∪ word:     0.8887
```

Word retrieval дополнительно нашёл 67 relevant items, пропущенных dense retriever.

### Character TF-IDF

```text
analyzer = char_wb
ngram_range = (3, 5)
max_features = 150_000
```

Он устойчивее к опечаткам и вариациям написания:

```text
FT2 ∪ word:        0.8887
+ char:             0.9005
```

### Description TF-IDF

Описание не добавлялось напрямую в E5, чтобы не перегружать embedding длинным рекламным текстом. Вместо этого сделан отдельный sparse channel:

```text
title + params + description[:1500]
```

Он дал самый заметный дополнительный прирост coverage:

```text
FT2 ∪ word ∪ char:       0.9005
+ description:            0.9335
```

Description спас 81 из 244 оставшихся misses.

---

## Почему location weight = 0.50

Проверялся score:

```text
semantic + alpha * location_prior
```

После `alpha ≈ 0.4` начинается плато:

| alpha | Recall@50 |
|---:|---:|
| 0.20 | 0.7606 |
| 0.30 | 0.7785 |
| 0.40 | 0.7851 |
| 0.50 | 0.7867 |
| 0.75 | 0.7863 |
| 1.00 | 0.7871 |

В финальном pipeline оставлено `0.50`: это сильный location signal, но он не доминирует над semantic similarity.

---

## Candidate union и top-300

Финальный candidate pool — объединение четырёх top-1000 retrieval channels.

Coverage полного union:

```text
2289 / 2452 = 93.35%
```

Перед CatBoost используется дешёвый score:

```text
semantic
+ 0.50 * location_prior
+ 0.005 * word_rank
```

Manual ranking:

| K | Recall |
|---:|---:|
| 50 | 0.8373 |
| 100 | 0.8764 |
| 200 | 0.9025 |
| 300 | 0.9139 |
| 500 | 0.9233 |

Поэтому выбрано `PRE_K=300`: coverage остаётся высоким, а число объектов для ML-reranker значительно уменьшается.

---

## CatBoostRanker

Для каждого из 300 кандидатов используются 17 признаков:

```text
semantic, location_prior, exact_location,
dense_rank, word_rank, char_rank, desc_rank,
in_dense, in_word, in_char, in_desc,
rating, log_reviews, log_price,
phone_hidden, message_forbidden,
query_has_params
```

Параметры:

```text
loss_function = YetiRank
iterations = 500
depth = 6
learning_rate = 0.05
l2_leaf_reg = 5
```

Большой hyperparameter search не проводился, чтобы не переобучаться на небольшой validation. Качество проверялось через 3-fold OOF.

Результаты:

```text
Manual Recall@50:   0.837276
CatBoost Recall@50: 0.853589
seen:               0.868798
unseen:             0.844660
```

CatBoost спас 67 запросов и потерял 27, net gain — `+40`.

---

## Итоговая архитектура

```text
E5 FT2 top-1000 ─────────────┐
Word TF-IDF top-1000 ────────┤
Char TF-IDF top-1000 ────────┼─> union, coverage 93.35%
Description TF-IDF top-1000 ─┘
                              │
                              ▼
             semantic + location + weak word rank
                              │
                              ▼
                     top-300, coverage 91.39%
                              │
                              ▼
                    CatBoostRanker
                              │
                              ▼
                           top-50
```

Главная идея решения — разделить **candidate recall** и **final ranking**. Retrieval-каналы компенсируют ошибки друг друга, а CatBoost уже выбирает лучшие 50 из сильных 300 кандидатов.

---

## Запуск

### Быстрое воспроизведение

Для генерации `answer.csv` повторный fine-tuning E5 не требуется.

Нужны:

```text
dataset/
e5_avito_pilot2_final/
artifacts/
candidate generation.ipynb
```

Запустить все ячейки `candidate generation.ipynb`.

Самый большой cache-файл `item_embeddings_e5_avito_pilot2.npy` не хранится в Git и при необходимости пересчитывается из E5 checkpoint.

### Полное воспроизведение

```text
1. train_e5.ipynb
2. получение e5_avito_pilot2_final/
3. candidate generation.ipynb
```

---

## Зависимости

```text
Python 3.12
numpy
pandas
pyarrow
torch
sentence-transformers
transformers
datasets
accelerate
scikit-learn
catboost
jupyterlab
```

На Apple Silicon используется MPS; при его отсутствии возможен CUDA/CPU fallback.
