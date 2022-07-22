import streamlit as st
import tensorflow as tf
from labels import LABELS

def predict_target(model, img, token=True):
    if token:
        img = tf.io.read_file(img)
    img = tf.image.decode_image(img, channels=3)
    im = tf.cast(tf.image.resize(img, size=[224, 224]), dtype=tf.float32)
    pred = tf.squeeze(model.predict(tf.expand_dims(im, axis=0)))
    result = LABELS[pred.argmax()]
    return im.numpy()/255., result


def main():
    st.set_page_config(
        page_title="Futurimo VA.DEV demo",
        page_icon="🧊"
    )

    model = tf.keras.models.load_model('model.h5')

    st.subheader('Try your own images')
    file = st.file_uploader(label='Your own image')
    if file is not None:
        original, pred = predict_target(model, file.getvalue(), token=False)
        col1, col2 = st.columns(2)
        col1.image(original, use_column_width='always', caption='Your original image')
        col2.text(pred, caption='AI title')


if __name__ == '__main__':
    main()