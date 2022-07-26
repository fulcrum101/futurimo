import streamlit as st
import tensorflow as tf
import numpy as np
from labels import LABELS

def predict_target(model, img, token=True):
    if token:
        img = tf.io.read_file(img)
    img = tf.image.decode_image(img, channels=3)
    im = tf.cast(tf.image.resize(img, size=[224, 224]), dtype=tf.float32)
    pred = tf.squeeze(model.predict(tf.expand_dims(im, axis=0)))
    result = LABELS[np.argmax(pred)]
    return im.numpy()/255., result


def main():
    st.set_page_config(
        page_title="Futurimo VA.DEV demo",
        page_icon="🧊"
    )

    model = tf.keras.models.load_model('model.h5')
    st.header('Futurimo VA.DEV demo')
    st.subheader('Try your own images')
    file = st.file_uploader(label='Your own image')
    if file is not None:
        original, pred = predict_target(model, file.getvalue(), token=False)
        st.image(original, use_column_width='always', caption='Your original image')
        st.text(f"AI Given Label: {pred}")


if __name__ == '__main__':
    main()