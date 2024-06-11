import gradio as gr
from fastai.vision.all import *

learner = load_learner('skin_cancer_detection.pkl');

def classify_image(img):
    pred, idx, probs = learner.predict(img)
    return dict(zip(learner.dls.vocab, map(float, probs)))

image = gr.Image();
label = gr.Label();

iface = gr.Interface(fn=classify_image, inputs=image, outputs=label)
iface.launch(inline=False)