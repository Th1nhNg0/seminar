# [submission_06_11_2022.json](submission_06_11_2022.json)

Model 1:

```py
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 128, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()
```

Model 2: bert-base-multilingual-cased

get top k document by TfidfVectorizer

Normalizing text tu raw data:

```py
def normalize_text(text):
    text = strip_html_tags(text)
    # remove \n
    text = re.sub(r'\n', ' ', text)
    # remove \xa0
    text = re.sub(r'\xa0', ' ', text)
    # remove 'BULLET::::-'
    text = re.sub(r'BULLET::::-', ' ', text)
    # remove = if more than 1
    text = re.sub(r'={2,}', ' ', text)
    # remove duplicate spaces
    text = re.sub(r'\s+', ' ', text)

    return text
```
