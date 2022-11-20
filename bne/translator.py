import os
import torch
import shutil

from .utils import FileLinks, download_file, spm_export_vocab, spm_encode, spm_decode

DEFAULT_WEIGHT_PATH = 'weights'
B2E_MODEL = os.path.join(DEFAULT_WEIGHT_PATH, FileLinks.base_bn2en[2])
E2B_MODEL = os.path.join(DEFAULT_WEIGHT_PATH, FileLinks.base_en2bn[2])
BN_MODEL = os.path.join(DEFAULT_WEIGHT_PATH, FileLinks.bn_model[2])
EN_MODEL = os.path.join(DEFAULT_WEIGHT_PATH, FileLinks.en_model[2])

def weightsDL( b2e ):
    if not os.path.exists(DEFAULT_WEIGHT_PATH):
        os.mkdir(DEFAULT_WEIGHT_PATH)
    if b2e == True:
        if not os.path.isfile(B2E_MODEL):
            download_file(FileLinks.base_bn2en[0], FileLinks.base_bn2en[1], B2E_MODEL)
    else:
        if not os.path.isfile(E2B_MODEL):
            download_file(FileLinks.base_en2bn[0], FileLinks.base_en2bn[1], E2B_MODEL)

    if not os.path.isfile(BN_MODEL):
        download_file(FileLinks.bn_model[0], FileLinks.bn_model[1], BN_MODEL) 

    if not os.path.isfile(EN_MODEL):
        download_file(FileLinks.en_model[0], FileLinks.en_model[1], EN_MODEL) 


def spmModel2Vocab(model_path):
    # model_prefix = model_path.split(os.path.sep)[-1].split('.')[0]
    # temp_dir = os.path.sep.join(model_path.split(os.path.sep)[:-1])
    
    spm_export_vocab(model_path)
    # vocab_cmd = [
    #     "spm_export_vocab --model",
    #     model_path,
    #     "| tail -n +4 >",
    #     model_path.replace('.model', '.vocab')
    # ]
    # os.system(" ".join(vocab_cmd))
    print(f"{ model_path.replace('.model', '.vocab') } generated.")


def spmOperate(CFG, temp_dir, tokenize):
    if tokenize:
        modelName = os.path.join(temp_dir, f"srcSPM.model")
        input_file = CFG.input_txt
        srcTok_path = f"{os.path.join(temp_dir, 'srctxt')}.tok"

        spm_encode( model_path= modelName, 
                    input_txt= input_file , 
                    input_tok= srcTok_path)
        # spm_cmd = [
        #     f"spm_encode --model=\"{modelName}\"",
        #     f"--output_format=piece",
        #     f"< \"{input_file}\" > \"{os.path.join(temp_dir, 'srctxt')}.tok\""
        # ]

        # os.system(" ".join(spm_cmd))
        print('Tokenization Completed')
        return srcTok_path
           

    else:
        tgtTok_path = f"{os.path.join(temp_dir, 'tgttxt')}.tok"
        output_file = os.path.join(CFG.output_dir, CFG.output_filename)

        modelName = os.path.join(temp_dir, f"tgtSPM.model")
        spm_decode(model_path=modelName, output_tok=tgtTok_path, output_txt = output_file)
        # spm_cmd = [
        #     f"spm_decode --model=\"{modelName}\"",
        #     f"< \"{tgtTok_path}\" > \"{output_file}\""
        # ]
        # os.system(" ".join(spm_cmd))
        
        # post_cmd = f"""sed 's/▁/ /g;s/  */ /g' -i \"{output_file}\""""
        # os.system(post_cmd)
        print('Detokenization Completed')
        return output_file
  

def translate(CFG, srcTok_path):
    tgtTok_path = srcTok_path.replace('srctxt', 'tgttxt')
    cmd = f'''
        onmt_translate \
            -model \"{CFG.src2tgt_model}\" \
            -src \"{srcTok_path}\" \
            -output \"{tgtTok_path}\" \
            -replace_unk -verbose -max_length {CFG.tgt_seq_length} -batch_size {CFG.eval_batch_size} {'-gpu 0' if torch.cuda.is_available() else ''}
    '''
    print(cmd)
    print('Translation Starting...')
    os.system(cmd)

    print('Translation Completed. Detokenizing...')
    return tgtTok_path



def bne_translate( b2e, src_sentence = None, src_textfile = None):
    
    weightsDL( b2e )
    class CFG:
        src2tgt_model = B2E_MODEL if b2e else E2B_MODEL
        src_lang_model = BN_MODEL if b2e else EN_MODEL
        tgt_lang_model = EN_MODEL if b2e else BN_MODEL
        output_dir = '.'

    if (src_sentence and src_textfile) or not (src_sentence or src_textfile):
        raise Exception("Provide one of src_sentence or src_textfile")

    temp_dir = os.path.join(CFG.output_dir, 'temp')

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    if src_sentence:
        src_textfile = os.path.join(temp_dir, 'src_textfile.txt')
        with open( src_textfile, 'w', encoding="utf-8") as writer:
            writer.write(src_sentence)
    print(src_textfile)
    CFG.input_txt = src_textfile
    CFG.output_filename = os.path.join('temp', 'preds.txt') if src_sentence else 'preds.txt'
    CFG.tgt_seq_length = 1024
    CFG.eval_batch_size = 8
    
    srcModel_tempPath = os.path.join(temp_dir, "srcSPM.model")
    shutil.copy(CFG.src_lang_model, srcModel_tempPath)

    tgtModel_tempPath = os.path.join(temp_dir, "tgtSPM.model")
    shutil.copy(CFG.tgt_lang_model, tgtModel_tempPath)

    srcVocab_tempPath = spmModel2Vocab(srcModel_tempPath)
    tgtVocab_tempPath = spmModel2Vocab(tgtModel_tempPath)

    srcTok_tempPath = spmOperate(CFG, temp_dir, tokenize = True)

    tgtTok_tempPath = translate(CFG, srcTok_tempPath)

    tgt_textfile = spmOperate(CFG, temp_dir, tokenize = False)

    with open(tgt_textfile, encoding="utf-8") as reader:
        tgt_sentence = reader.read()
    
    return tgt_sentence


if __name__ == "__main__":
    b2e = True
    s = bne_translate(b2e, src_sentence = 'সেরা আন্তর্জাতিক সিনেমা বিভাগে ছবিটি মনোনীত হয়েছে।')
    print(s)