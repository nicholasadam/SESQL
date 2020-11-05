# Wangsu Hu
# Oct 2020
import argparse, json, torch
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel
gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_hyper_param(parser):
	parser.add_argument("--bS", default=32, type=int, help="Batch size")
	parser.add_argument("--toy", default=False, help="toy samples")

	args = parser.parse_args()
	return args


def get_data(path_wikisql, mode, bS, toy_model = False, toy_size = 12):
	path_sql = path_wikisql + mode +'_tok.jsonl'
	path_table = path_wikisql + mode + '.tables.jsonl'
	data = []
	table = {}
	with open(path_sql) as f:
		for idx, line in enumerate(f):
			if toy_model and idx >= toy_size:
				break
			t1 = json.loads(line.strip())
			data.append(t1)
	with open(path_table) as f:
		for idx, line in enumerate(f):
			if toy_model and idx > toy_size:
				break
			t1 = json.loads(line.strip())
			table[t1['id']] = t1
	loader = torch.utils.data.DataLoader(
		batch_size=bS,
		dataset=data,
		shuffle= mode == 'train',
		num_workers=4,
		collate_fn=lambda x: x  
	)
	return data, table, loader

def get_bert(path_bert):
    bert_config_file = path_bert + 'bert_config_uncased_L-12_H-768_A-12.json'
    vocab_file = path_bert + 'vocab_uncased_L-12_H-768_A-12.txt'
    init_checkpoint = path_bert + 'pytorch_model_uncased_L-12_H-768_A-12.bin'
    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    model_bert = BertModel(bert_config)    
    model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
    print("Load pre-trained parameters.")
    if gpu:
	    model_bert.to(device)
    return model_bert, tokenizer, bert_config

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = construct_hyper_param(parser) 
	path_wikisql = './wikisql/'  
	path_bert = './bert/'

	path_save_for_evaluation = './'
	train_data, train_table, train_loader = get_data(path_wikisql, 'train', args.bS, args.toy)
	model_bert, tokenizer, bert_config = get_bert(path_bert)

	dev_data, dev_table, train_loader = get_data(path_wikisql, 'dev', args.bS)


	test_data, test_table, test_loader = get_data(path_wikisql, 'test', args.bS)
