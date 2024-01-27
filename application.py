# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import transformers
# from transformers import AutoModel, BertTokenizerFast

# with open('tokenizer_bert.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)
# with open('spam_classifier_bert.pkl', 'rb') as f:
#     model = pickle.load(f)

# # tokenizer = pickle.load(open('tokenizer_bert.pkl', 'rb'))
# # model = pickle.load(open('spam_classifier_bert.pkl', 'rb'))

# def get_embeddings(text, tokenizer):
#     tokens = tokenizer.batch_encode_plus(
#         text.tolist(),
#         max_length = 25,
#         pad_to_max_length=True,
#         truncation=True,
#         return_token_type_ids=False
#     )
#     text_seq = torch.tensor(tokens['input_ids'])
#     text_mask = torch.tensor(tokens['attention_mask'])
    
#     return text_seq, text_mask

# # if __name__ == '__main__':

# st.header('Spam Detection')

# text = st.text_input('Enter Text')
# text_seq, text_mask = get_embeddings(text, tokenizer)


# if st.button('Check Spam'):
#     with torch.no_grad():
#         pred = model(text_seq, text_mask)
#         pred = pred.detach().cpu().numpy()


# result = np.argmax(pred, axis = 1)
# if result:
#     st.header('Spam')
# else:
#     st.header('Non-Spam')
    
    

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

# specify GPU
# device = torch.device("cuda")

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', return_dict=False)
# Load spam classifier model
# with open('model.pkl', 'rb') as f:

model = torch.load('model.h5',map_location=torch.device('cpu'))

# Function to classify text as spam or not
def classify_text(text):
    # Tokenize input text
    tokens_test = tokenizer.batch_encode_plus(
                        [text],
                        max_length = 25,
                        pad_to_max_length=True,
                        truncation=True,
                        return_token_type_ids=False
                    )
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    # test_y = torch.tensor(test_labels.tolist())
    # get predictions for test data
    with torch.no_grad():
        preds = model(test_seq, test_mask)
        # preds = preds.detach().cpu().numpy()

    # Get predicted label
    predicted_label = np.argmax(preds, axis = 1).tolist()[0]

    return predicted_label

# Streamlit app
st.title("Spam Classifier App")

# User input
user_input = st.text_area("Enter text:")

# Make prediction when user clicks the button
if st.button("Classify"):
    if user_input:
        # Classify text
        prediction = classify_text(user_input)

        # Display the result
        if prediction == 1:
            st.error("Spam Detected!")
        else:
            st.success("Not Spam.")
    else:
        st.warning("Please enter some text.")
