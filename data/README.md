# data/

This folder holds **instructions and code to get the data — not the data itself.**

The actual CSVs live on your disk under `data/` but are **ignored by git**
(see the repo-root `.gitignore`). Anyone cloning the repo runs the scripts here
to reproduce the dataset locally.

## Dataset

**Customer Support on Twitter** — ~3M tweets between users and brand support
accounts.

- Source: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Main file: `archive/twcs/twcs.csv` (~493 MB, 3,002,523 rows)

### Columns

| column                    | description                                             |
|---------------------------|---------------------------------------------------------|
| `tweet_id`                | unique id for the tweet                                 |
| `author_id`               | anonymized author (user handle or brand)                |
| `inbound`                 | `True` if the tweet is from a customer to a brand       |
| `created_at`              | timestamp the tweet was sent                            |
| `text`                    | tweet body                                              |
| `response_tweet_id`       | id(s) of tweet(s) that responded to this one            |
| `in_response_to_tweet_id` | id of the tweet this one is responding to               |

## Getting the data

```bash
# 1. Download the raw dataset (needs Kaggle CLI + ~/.kaggle/kaggle.json)
bash data/download.sh

# 2. Produce the cleaned table -> data/twcs_clean.csv
python data/preprocess.py
```

## Layout

```
data/
├── download.sh        # committed — fetches raw data from Kaggle
├── preprocess.py      # committed — cleans raw CSV into twcs_clean.csv
├── README.md          # committed — this file
├── archive/           # NOT committed — raw download lands here
│   └── twcs/twcs.csv
└── twcs_clean.csv     # NOT committed — output of preprocess.py
```

## What NOT to commit

Never commit the CSVs (or `.zip` / `.parquet` / the `archive/` folder). They are
large and reproducible from the scripts above. The `.gitignore` already excludes
them — keep it that way.
