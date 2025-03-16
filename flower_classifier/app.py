from fastai.vision.all import * 
import gradio as gr

learner = load_learner('models/first_classifier.pkl')

categories = learner.dls.vocab

def classify_image(img) : 
    pred, idx, probs = learner.predict(img)
    return(dict(zip(categories, map(float, probs))))

title = "My first flower classifier"
description = "This classifier can handle 5 types of flowers : primroses, roses, tulips, orchids and geraniums."
image = gr.Image()
label = gr.Label()
examples = ['primrose.jpg']

intf = gr.Interface(fn=classify_image, 
                    title = title, description = description,
                    inputs=image, outputs=label, 
                    examples=examples)
intf.launch(share=True)