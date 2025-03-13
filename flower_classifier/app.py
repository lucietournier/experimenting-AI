from fastai.vision.all import * 
import gradio as gr

learner = load_learner('flower_classifier/models/first_classifier.pkl')

categories = learner.dls.vocab

def classify_image(img) : 
    pred, idx, probs = learner.predict(img)
    return(dict(zip(categories, map(float, probs))))

image = gr.Image()
label = gr.Label()
examples = ['primrose.lpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)