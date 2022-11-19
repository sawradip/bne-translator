import wget
import gdown
import sentencepiece as spm

class FileLinks:
    bn_model = ('url', 
                'https://github.com/sawradip/Bang2Eng/blob/main/bn.model?raw=true', 
                'bn.model')

    en_model = ('url', 
                'https://github.com/sawradip/Bang2Eng/blob/main/en.model?raw=true', 
                'en.model') 

    base_bn2en = (  'gdrive',
                    '1xx5bU31sIMU24kLm5bYh19qLBYHHriP4',
                    'base_bn2en.pt')

    base_en2bn = (  'gdrive',
                    '1-RLTuuOvSPB1Qzmho9WFlhAYnbYxPI_Q'
                    'base_en2bn.pt')

def download_file(link_type, link, file_destination):
    if link_type == 'url':
        wget.download(link, file_destination)
    elif link_type == 'gdrive':
        gdown.download(id=link, output=file_destination)
    else:
        raise Exception('Only url and gdrive are suppoted.')

        
def spm_export_vocab(model_file, vocab_file = None):
    vocab_dict = {}
    if vocab_file is None:
        vocab_file = model_file.replace(".model", ".vocab")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    vocabs = [sp.IdToPiece(id) for id in range(sp.GetPieceSize())]
    with open(vocab_file, "w") as vfile:
        for v in vocabs:
            id = sp.PieceToId(v)
            vocab_dict[v] = sp.GetScore(id)
            vfile.write(f'{v}\t{sp.GetScore(id)}\n')

    return vocab_dict

def spm_encode(model_path, input_txt, input_tok):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    with open(input_txt, "r") as tfile:
        lines = tfile.readlines()

    tok_lines = []
    for line in lines:
        ids = sp.EncodeAsIds(line)
        tok_line = " ".join(sp.IdToPiece(ids))
        tok_lines.append(tok_line)

    tok_lines = "\n".join(tok_lines)
    with open(input_tok, "w") as tfile:
        tfile.write(tok_lines)

def spm_decode(model_path, output_tok, output_txt):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    with open(output_tok, "r") as tfile:
        tok_lines = tfile.readlines()

    lines = []
    for line in tok_lines:
        ids = sp.EncodeAsIds(line)
        line = sp.DecodeIds(ids)
        lines.append(line)

    lines = "\n".join(lines)
    with open(output_txt, "w") as tfile:
        tfile.write(lines)