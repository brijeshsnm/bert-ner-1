class NERModel(nn.Module):
    def __init__(self, num_tag):
        
        super(NERModel, self).__init__()
        self.num_tag = num_tag
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.bert_drop = nn.Dropout(0.3)    
        self.out_tag_in = nn.Linear(768, 1000)  
        self.out_tag = nn.Linear(1000, self.num_tag)  
  
    def forward(self, token_ids, token_type_ids=None, attention_mask=None):
        sequence_output = self.bert(token_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)[0]        
        sequence_output = self.bert_drop(sequence_output)
        sequence_output = self.out_tag_in(sequence_output)
        logits = self.out_tag(sequence_output)
        return logits

model = NERModel(n_tags)
model.load_state_dict(torch.load('E:\\NERData\\model\\model_original_5'))
model = model.to(device)

import spacy
nlp = spacy.load("en_core_web_lg")

def tokenize_raw_sentences(sentences):
    sent_id = 1
    token_list = []    
    attention_mask_list = []
    token_type_list = []
    target_mask_list = []
    stem_list = []
    
    for sent in sentences: 
        words = sent.split()
        df = pd.DataFrame(words, columns = ['token'])
        df['sentence_id'] = 1
        df['label'] = ''
        sent_tokens = tokenize_sentences(df, bert_tokenizer, 64)

        token_ids = sent_tokens['token_id'].values.tolist()
        attention_masks = sent_tokens['attention_mask'].values.tolist()
        target_mask = sent_tokens['target_mask'].values.tolist()  
        stem = sent_tokens['stem'].values.tolist()  
        token_type_ids = [0] * len(attention_masks)  
        
        token_list.append(token_ids)
        attention_mask_list.append(attention_masks)
        token_type_list.append(token_type_ids)
        target_mask_list.append(target_mask)
        stem_list.append(stem)
        
    
    return {          
          'token_ids':torch.tensor(token_list, dtype=torch.long),
          'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
          'token_type_ids': torch.tensor(token_type_list, dtype=torch.long),
          'target_mask': torch.tensor(target_mask_list, dtype=torch.long),
          'stem': stem_list
        }

raw_txt = """
"""


doc = nlp(raw_txt)
sentences = []
for sent in doc.sents:
    sent = sent.string.strip()
    sent = sent.replace("(", " ( ")
    sent = sent.replace(")", " )")
    sent = sent.replace("/", " / ")
    sent = sent.replace(",", " ,")
    sent = sent.replace("'", " '")
    sent = sent.replace("’", " ’")
    
    
    sentences.append(sent)
#sentences = [sent.string.strip() for sent in doc.sents]

data = tokenize_raw_sentences(sentences)

with torch.no_grad():
    
    input_ids = data["token_ids"].to(device)
    token_type_ids = data["token_type_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    target_mask = data["target_mask"]
    stem = data["stem"]
   
    logits = model(input_ids, token_type_ids, attention_mask) 
    #print(logits)
    logits = torch.nn.functional.softmax(logits, dim = 2)
    pred_val, preds = torch.max(logits, dim=2)
    preds = preds.cpu().numpy()
    pred_val = pred_val.cpu().numpy()
    target_valid =  target_mask.cpu().numpy() == 1 
    output_list = []
    
    for i in range(len(sentences)):       
        
        target_valid_sent = target_valid[i]
        pred_sent = preds[i]
        pred_val_sent = pred_val[i]
        preds_flatten = pred_sent[target_valid_sent]
        preds_val_flatten = pred_val_sent[target_valid_sent]
        stem_sent = stem[i]
        pred_tags = enc_tag.inverse_transform(preds_flatten).tolist()
        input_words = list(compress(stem_sent, target_valid_sent))
        output = zip(input_words, pred_tags, preds_val_flatten)
        output_list.append(list(output)[1:-1])
    

prev_sent = ''
with open('E:\\NERData\\ADHOC\\ADHOC_WEB.txt', 'w') as f1:    
    for sent in output_list:   
        f1.write("\n")
        for word in sent:            
            try: 
                f1.write(f'{word[0]} {word[1]} \n')               
            except:
                pass
