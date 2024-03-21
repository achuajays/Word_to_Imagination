from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, render_template, request


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("2003achu/imagination_of_word")
model = AutoModelForSeq2SeqLM.from_pretrained("2003achu/imagination_of_word")



app = Flask(__name__)

def generate(prompt):

    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    word = None
    value = None
    if request.method == 'POST':
        word = request.form['word']
        value = generate(word)
        return render_template('index.html', word=word, value=value)
    return render_template('index.html', word=word, value=value)

if __name__ == '__main__':
    app.run(debug=True)