# Avito Candidate Generation

Решение тестового задания по генерации кандидатов для поиска в категории услуг.

Для каждого поискового запроса требуется выбрать до 50 объявлений из общего корпуса.  
Основная цель — максимизировать `Recall@50`.

## Структура проекта

```text
.
├── candidate_generation.ipynb      # основной pipeline и генерация answer.csv
├── train_e5.ipynb                  # двухэтапный fine-tuning multilingual E5
├── answer.csv                      # итоговый файл с предсказаниями
│
├── e5_avito_pilot2_final/          # готовый checkpoint финальной E5-модели
│
├── artifacts/                      # предрассчитанные embeddings и retrieval-кэши
│   ├── item_embeddings_e5_avito_pilot2.npy 
│   ├── val_embeddings_e5_avito_pilot2.npy
│   ├── benchmark_query_embeddings_ft2.npy
│   ├── dense_e5_avito_pilot2_indices_top1000.npy
│   ├── dense_e5_avito_pilot2_scores_top1000.npy
│   ├── benchmark_ft2_dense_indices_top1000.npy
│   ├── benchmark_ft2_dense_scores_top1000.npy
│   ├── lexical_tfidf_indices_top1000.npy
│   ├── benchmark_lexical_indices_top1000.npy
│   ├── char_tfidf_indices_top1000.npy
│   ├── benchmark_char_tfidf_indices_top1000.npy
│   ├── description_tfidf_indices_top1000.npy
│   ├── benchmark_description_tfidf_indices_top1000.npy
│   └── catboost_ranker_final.cbm
└── 
    
```

## Общая схема решения

Решение состоит из двух этапов:

1. генерация большого набора кандидатов несколькими независимыми retrieval-моделями;
2. отбор финальных 50 кандидатов с помощью `CatBoostRanker`.

Используются четыре retrieval-канала:

- fine-tuned `intfloat/multilingual-e5-small`;
- word-level TF-IDF;
- character-level TF-IDF;
- word-level TF-IDF с добавлением текста описания объявления.

Для каждого канала извлекается до 1000 кандидатов. Затем их объединение дополнительно оценивается по semantic similarity и location prior, после чего предварительно отбираются top-300 кандидатов. Финальные top-50 выбираются CatBoost ranker.

## Валидация

Для локальной оценки была сформирована validation-выборка из 2452 запросов:

- 907 `seen` запросов — query text встречается в обучающей выборке, но соответствующая hidden query-item пара удаляется;
- 1545 `unseen` запросов — query text полностью исключается из обучающей выборки.

Relevant item выбирается только среди объявлений, присутствующих в `benchmark_items.parquet`.

Такая схема позволяет отдельно оценивать качество на уже встречавшихся и новых формулировках запросов и не использовать скрытые positive-пары при обучении.

## Fine-tuning E5

Dense retriever основан на:

```text
intfloat/multilingual-e5-small
```

Текстовые представления:

```text
query: <search_query>. <search_infm_params_text>
passage: <item_title_raw>. <item_infm_params_text>
```

Описание объявления в E5 не используется и обрабатывается отдельным TF-IDF каналом.

Fine-tuning выполняется в `train_e5.ipynb` в два этапа.

### FT1

Для каждого уникального query выбирается одна positive query-item пара.

Основные параметры:

```text
loss: MultipleNegativesRankingLoss
epochs: 1
batch size: 32
learning rate: 2e-5
max sequence length: 128
batch sampler: NO_DUPLICATES
```

После первого этапа сохраняется промежуточный checkpoint:

```text
e5_avito_pilot_final/
```

### FT2

На втором этапе обучение продолжается с checkpoint FT1.

Для запросов с несколькими positive items по возможности выбирается другой positive item. В исходном обучении альтернативный positive удалось использовать для 36 112 из 104 063 запросов.

Параметры:

```text
loss: MultipleNegativesRankingLoss
epochs: 1
batch size: 32
learning rate: 1e-5
max sequence length: 128
batch sampler: NO_DUPLICATES
```

Итоговый checkpoint:

```text
e5_avito_pilot2_final/
```

Именно он используется в основном pipeline.

## Candidate retrieval

### Dense retrieval

Для всех объявлений и запросов вычисляются L2-нормализованные E5 embeddings размерности 384.

Cosine similarity вычисляется как скалярное произведение нормализованных векторов. Для каждого запроса выбираются top-1000 объявлений.

### Word TF-IDF

Используемый текст:

```text
query: search_query + search_infm_params_text
item: item_title_raw + item_infm_params_text
```

Параметры:

```text
analyzer = word
ngram_range = (1, 2)
min_df = 2
max_features = 200000
sublinear_tf = True
```

### Character TF-IDF

Используется для устойчивости к опечаткам, вариациям написания и редким терминам.

```text
analyzer = char_wb
ngram_range = (3, 5)
min_df = 2
max_features = 150000
```

### Description TF-IDF

В item representation дополнительно добавляются первые 1500 символов `item_description_raw`.

```text
item = title + params + description[:1500]
query = search_query + search_infm_params_text
```

## Location prior

По обучающим данным оценивается:

```text
P(item_location_id | search_location_id)
```

Локация используется только как soft feature, а не как жёсткий фильтр.

Для локальной validation prior строится по leakage-safe train. Для benchmark используется весь доступный `train.parquet`.

## Объединение кандидатов

Top-1000 кандидатов четырёх retrieval-каналов объединяются.

Для каждого кандидата сохраняются:

- E5 semantic score;
- location prior;
- exact location match;
- rank в каждом retrieval-канале;
- индикатор присутствия кандидата в каждом retrieval-канале.

Для предварительного отбора используется score:

```text
semantic
+ 0.50 * location_prior
+ 0.005 * word_rank
```

По нему выбираются top-300 кандидатов.

## CatBoost ranker

Финальный `CatBoostRanker` переупорядочивает top-300 кандидатов.

Используются 17 признаков:

```text
semantic
location_prior
exact_location

dense_rank
word_rank
char_rank
desc_rank

in_dense
in_word
in_char
in_desc

rating
log_reviews
log_price

phone_hidden
message_forbidden

query_has_params
```

Параметры модели:

```text
loss_function = YetiRank
iterations = 500
depth = 6
learning_rate = 0.05
l2_leaf_reg = 5
random_seed = 42
```

На локальной OOF-валидации CatBoost улучшил `Recall@50` относительно ручного pre-ranking.

## Запуск

### Быстрое воспроизведение итогового результата

Для генерации `answer.csv` повторный fine-tuning E5 не требуется.

Необходимо наличие:

```text
dataset/ (в который необходимо поместить все исходные файлы .parquet)
candidate_generation.ipynb
```

После этого достаточно открыть:

```text
candidate_generation.ipynb
```

и выполнить все ячейки.

Если готовые `.npy` artifacts присутствуют, они загружаются из кэша. Если соответствующий кэш отсутствует, основной notebook может пересчитать необходимые представления из сохранённого E5 checkpoint.

### Полное воспроизведение обучения

Для повторного обучения dense retriever:

1. запустить `train_e5.ipynb`;
2. получить `e5_avito_pilot2_final/`;
3. запустить `candidate_generation.ipynb`.

Два этапа fine-tuning являются наиболее долгими частями pipeline.

## Формат результата

Итоговый файл:

```text
answer.csv
```

содержит два столбца:

```text
query_id,answer
```

где `answer` — строка из 50 `item_id`, разделённых пробелами.

## Зависимости

Основные библиотеки:

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

На Apple Silicon автоматически может использоваться `MPS`. При его отсутствии код может выполняться на CUDA или CPU.

## Примечание о воспроизводимости

Для быстрого и максимально точного воспроизведения submission используется сохранённый checkpoint `e5_avito_pilot2_final` и готовые artifacts.

Код полного обучения модели приведён отдельно в `train_e5.ipynb`. Повторное обучение E5 не требуется для генерации итогового `answer.csv`.