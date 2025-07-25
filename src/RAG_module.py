#pip install -qU transformers accelerate bitsandbytes langchain langchain-core langchain-community langchain-text-splitters langchain_huggingface faiss-cpu sentence_transformers
import torch
import json
import re
import os
import threading

from typing import Any, List, Union
from sentence_transformers import util, SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAG_answer_package_viettel:
    def __init__(self,
                 name_model_llm: str = "vilm/vinallama-7b-chat",
                 # the model name LLM main for answer the user question
                 name_model_embedding: str = "keepitreal/vietnamese-sbert",
                 # the model use for embedding context to vector this model save vector store and load vector store for main model"""
                 data_path_for_RAG: Union[str, List[str]] = None,
                 # data path (Json folder) but at there i have 2 folder need process
                 vector_store_path: str = None,
                 # folder path vector store saved by FAISS
                 llm: Any = None,
                 prompt_template: Any = None):

        self.name_model_llm = name_model_llm
        self.name_model_embedding = name_model_embedding
        self.data_path_for_RAG = data_path_for_RAG
        self.vector_store_path = vector_store_path

        self.llm = llm
        """the main model llm need define out come class"""
        self.prompt_template = prompt_template
        """the prompt"""
        self.model, self.tokenizer = self.get_model_tokenizer()

        self.model_embedding = HuggingFaceEmbeddings(
            model_name=self.name_model_embedding,
            model_kwargs={"device": device},
            show_progress=True
        )
        self.model_find_relevant_context = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        self.vector_store = None
        self.prompt = None
        # load vector store when build vector store done

    def build_vector_store(self):
        text_from_json = []

        def case_for_folder_one(folder_path):
            """if data path just only a folder
               compare each file json and get conversation
               the conversation struction in json like this
                {   "conversations" :
                    [
                        {
                            "role" : "user",
                            "content" : "Question",
                        },
                        {
                            "role" : "assistant",
                            "content" : "Answer",
                        }
                    ]
            }"""
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"DATA PATH JSON NOT FOUND PLEASE CHECK {folder_path}")

            for short_name_file_json in os.listdir(folder_path):
                file_path_json = os.path.join(folder_path, short_name_file_json)

                with open(file_path_json, "r", encoding="utf-8") as f:
                    contents_file_json = json.load(f)  # load json

                for content in contents_file_json:
                    a_conversation = str(
                        f'Q: {content["conversations"][0]["content"]}\nA: {content["conversations"][1]["content"]}')
                    """get content of conversation on json file"""
                    text_from_json.append(a_conversation)

        def case_for_folder_two(folder_path):
            """if data path have two folder
               compare each file json and get conversation or content
               the conversation struction in json like those
                case:2
                {
                    "text": "content",
                    "metadata":
                    {
                        "url": "URL package",
                        "title": "package name",
                        "section": "Ưu đãi"
                    }
                }
                """
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"DATA PATH JSON NOT FOUND PLEASE CHECK {folder_path}")

            for short_name_file_json in os.listdir(folder_path):
                file_path_json = os.path.join(folder_path, short_name_file_json)

                with open(file_path_json, "r", encoding="utf-8") as f:
                    contents_file_json = json.load(f)  # load json

                for i in range(len(contents_file_json)):
                    a_conten = f'{contents_file_json[i]["metadata"]["section"]} của gói cước {contents_file_json[i]["metadata"]["title"]} {contents_file_json[i]["text"]}'
                    """get content of conversation on json file"""
                    text_from_json.append(a_conten)

        if isinstance(self.data_path_for_RAG, str):
            case_for_folder_one(self.data_path_for_RAG)

        elif isinstance(self.data_path_for_RAG, list):
            case_for_folder_one(self.data_path_for_RAG[0])
            case_for_folder_two(self.data_path_for_RAG[1])
        documents = [Document(page_content=text) for text in text_from_json]
        vector_db = FAISS.from_documents(documents, self.model_embedding)
        vector_db.save_local(self.vector_store_path)

    def load_vector_store(self):
        vector_store = FAISS.load_local(self.vector_store_path,
                                        self.model_embedding,
                                        allow_dangerous_deserialization=True
                                        )
        return vector_store

    def create_prompt(self, flag, context, question):
        if flag == 1:
            self.prompt_template = f"""<|im_start|>system
Bạn là một trợ lý AI hữu ích tên là TUNGLLAMA. Hãy chỉ dựa vào thông tin trong ngữ cảnh bên dưới để trả lời câu hỏi một cách chính xác, ngắn gọn và không lặp lại. Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy trả lời "Tôi không tìm thấy thông tin trong ngữ cảnh".
<|im_end|>
<|im_start|>user
Ngữ cảnh:
{context}

Câu hỏi:
{question}
<|im_end|>
<|im_start|>assistant
"""
            return PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        elif flag == 0:
            self.prompt_template = f"""<|im_start|>system
Bạn là một trợ lí AI tên là TungLlama. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
            return PromptTemplate(template=self.prompt_template, input_variables=["question"])
        return None

    def get_model_tokenizer(self):
        # load model mode 4 bit
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.name_model_llm,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=nf4_config
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.name_model_llm,
            trust_remote_code=True
        )
        return model, tokenizer

    def QA_chain(self, flag, context, question):
        """create chain"""
        global qa_chain
        if flag == 1:
            self.prompt = self.create_prompt(flag, context, question)
            qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt
            )
        elif flag == 0:
            self.prompt = self.create_prompt(flag, context, question)
            qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt
            )
        return qa_chain

    def _get_context_and_clean(self, question_user: str, k_search_question: int, k_search_package: int,
                               ratio_question_package=0.4):
        """
            CASE 1: detected package name in user's question
                processed context relevant in 20 context finded by similarity_search with 2 cases
                *NOTE: AND HAVE NAME PACKAGE IN USER QUESTION
                docs 1: search with package name in vector store and get 10 context by similarity_search
                docs 2: search with question user in vector store and get 20 context by similarity_search
                then sum 2 documents
                then have 2 cases next
            ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                case 1.1: get vector store have conversation (in conversation have {Q: question,
                                                                                  A: answer})
                    case 1.1.1 compare user's question and question in context detected if match i have score compute by cos_sim (0-1)
                    case 1.1.2 if detected package name done -> compare package name in user's question and context finded if match i have score compute by cos_sim (0-1)
                    there I set the similarity match ratio between [(user question and context question)/(package name in user question)/(package name in context) = 0.4/0.6
                    */score_question_and_context*ratio_question_package + score_find_package_relevant*(1-ratio_question_package)/* ratio_question_package=0.4
            ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                case 1.2: have no conversation this have only infor about package
                    case 1.2.1 compare user's question and context finded if match i have score compute by cos_sim (0-1)
                    case 1.2.2 if detected package name done -> compare package name in user's question and package name in context finded if match i have score compute by cos_sim (0-1)
                    there I set the similarity match ratio between [(user question and context)/(package name in user question)/(package name in context) = 0.4/0.6
                    */score_question_and_context*ratio_question_package + score_find_package_relevant*(1-ratio_question_package)/* ratio_question_package=0.4
            CASE 2: Can't detect package name in user's question
                return flag =0 and user's question for model no context !
        """
        self.vector_store = self.load_vector_store()

        device = self.model_find_relevant_context.device

        _name_package_from_user_Q = re.findall(r"\b(?=\w{3,30}\b)(?=\w*\d)(?=\w*[a-zA-Z])[a-zA-Z0-9_]+\b",
                                               question_user)
        # print(_name_package_from_user_Q)
        _name_package_from_user_Q = [match.upper() for match in
                                     _name_package_from_user_Q]  # ex user input 2 package (pass)
        if len(_name_package_from_user_Q) > 0:
            question_user_encode = self.model_find_relevant_context.encode(question_user,
                                                                           convert_to_tensor=True,
                                                                           batch_size=16,
                                                                           show_progress_bar=False).to(device)

            _name_package_from_user_Q_encode = self.model_find_relevant_context.encode(_name_package_from_user_Q[0],
                                                                                       convert_to_tensor=True,
                                                                                       batch_size=16,
                                                                                       show_progress_bar=False).to(
                device)
            # search the 20 content relevant
            documents1 = self.vector_store.similarity_search(question_user, k=k_search_question)
            documents2 = self.vector_store.similarity_search(_name_package_from_user_Q[0], k=k_search_package)
            documents = documents1 + documents2
            # array to save score and answer corresponding
            score_answer = []
            # context raw get from search
            contexts_raw = "\n\n".join([doc.page_content for doc in documents])
            # print(contexts_raw)
            contexts_raw = contexts_raw.strip().split("\n\n")
            # v2 /*
            Q_parts = []
            name_packages_Q = []
            contexts_case2 = []
            name_packages_case2 = []

            indexes_case1 = []
            indexes_case2 = []  # v2 */

            for i, context in enumerate(contexts_raw):  # iterate through each of the 20 searched contexts
                if "Q:" in context and "A:" in context:  # case 1: conversation have Q: question and A: answer
                    Q_part_in_context = context.split("A: ")[0].replace("Q: ", "").replace("\n",
                                                                                           "")  # split question and answer
                    A_part_in_context = context.split("A: ")[1]
                    """
                    # old version (v1)
                        #compare Q in context with question user if score high get the best
                        # score_question = util.cos_sim(question_user_encode,
                        #                              self.model_find_relevant_context.encode(Q_part_in_context, 
                        #                                                 convert_to_tensor=True,
                        #                                                 batch_size=16,
                        #                                                 show_progress_bar=False).to(device))[0][0].item()"""
                    Q_parts.append(Q_part_in_context)

                    # get name package in question context
                    _name_package_from_context_Q = re.findall(
                        r"\b(?=\w{3,30}\b)(?=\w*\d)(?=\w*[a-zA-Z])[a-zA-Z0-9_]+\b", Q_part_in_context)
                    # print(_name_package_from_context_Q)
                    _name_package_from_context_Q = [match.upper() for match in _name_package_from_context_Q]

                    name_packages_Q.append(_name_package_from_context_Q)

                    indexes_case1.append((i, A_part_in_context))
                    """
                    old_version
                    if len(_name_package_from_context_Q)>0:
                        #compare question user about package A with Question context about package A 
                        score_find_package_relevant = util.cos_sim(_name_package_from_user_Q_encode,
                                                      self.model_find_relevant_context.encode(_name_package_from_context_Q[0], 
                                                                        convert_to_tensor=True,
                                                                        batch_size=16,
                                                                        show_progress_bar=False).to(device))[0][0].item()
                        #Calculate average score between score_question(40%) and score_package(60%)
                        average_score_case1=score_question*ratio_question_package + score_find_package_relevant*(1-ratio_question_package)
                        # print("case 1: ",average_score_case1)
                        score_answer.append((average_score_case1,A_part_in_context))

                    else:
                        # Here are some cases where the package name cannot be detected.
                        score_find_package_relevant = 0
                        #Calculate average score between score_question(40%) and score_package(60%)
                        average_score_case1=score_question*ratio_question_package + score_find_package_relevant*(1-ratio_question_package)
                        # print("case 1: ",average_score_case1)
                        score_answer.append((average_score_case1,A_part_in_context))"""

                else:  # case2: have no conversation this have only infor about package

                    # v2
                    contexts_case2.append(context)
                    """old version
                    # score_question_and_context = util.cos_sim(question_user_encode,
                    #                               self.model_find_relevant_context.encode(context, 
                    #                                                 convert_to_tensor=True,
                    #                                                 batch_size=16,
                    #                                                 show_progress_bar=False).to(device))[0][0].item()
                    #get name package in context """
                    _name_package_from_context = re.findall(r"\b(?=\w{3,30}\b)(?=\w*\d)(?=\w*[a-zA-Z])[a-zA-Z0-9_]+\b",
                                                            context)

                    _name_package_from_context = [match.upper() for match in _name_package_from_context]  ########

                    # v2
                    name_packages_case2.append(_name_package_from_context)

                    indexes_case2.append(i)  # v2
                    """
                    this old_version
                    if len(_name_package_from_context)>0:
                        #compare question user about package A with context about package A 
                        score_find_package_relevant = util.cos_sim(_name_package_from_user_Q_encode,
                                                                  self.model_find_relevant_context.encode(_name_package_from_context[0], 
                                                                        convert_to_tensor=True,
                                                                        batch_size=16,                                  
                                                                        show_progress_bar=False).to(device))[0][0].item()
                        #Calculate average score between score_question_and_context(40%) and score_package(60%)
                        average_score_case2=score_question_and_context*ratio_question_package + score_find_package_relevant*(1-ratio_question_package)
                        # print(context)
                        # print("case 2:",average_score_case2)
                        score_answer.append((average_score_case2,context))
                    else:
                        # Here are some cases where the package name cannot be detected.
                        score_find_package_relevant = 0
                        #Calculate average score between score_question(40%) and score_package(60%)
                        average_score_case1=score_question*ratio_question_package + score_find_package_relevant*(1-ratio_question_package)
                        # print("case 1: ",average_score_case1)
                        score_answer.append((average_score_case1,A_part_in_context))"""

            # v2 /*
            Q_encodes = self.model_find_relevant_context.encode(Q_parts,
                                                                convert_to_tensor=True,
                                                                batch_size=16,
                                                                show_progress_bar=False).to(device)

            flat_name_packages_Q = [x[0] if len(x) > 0 else "" for x in name_packages_Q]
            package_encodes_case1 = self.model_find_relevant_context.encode(flat_name_packages_Q,
                                                                            convert_to_tensor=True,
                                                                            batch_size=16,
                                                                            show_progress_bar=False).to(device)

            contexts_case2_encodes = self.model_find_relevant_context.encode(contexts_case2,
                                                                             convert_to_tensor=True,
                                                                             batch_size=16,
                                                                             show_progress_bar=False).to(device)

            flat_name_packages_case2 = [x[0] if len(x) > 0 else "" for x in name_packages_case2]
            package_encodes_case2 = self.model_find_relevant_context.encode(flat_name_packages_case2,
                                                                            convert_to_tensor=True,
                                                                            batch_size=16,
                                                                            show_progress_bar=False).to(device)

            # compute score for case1:
            for idx, (i, A_part) in enumerate(indexes_case1):
                score_question = util.cos_sim(question_user_encode, Q_encodes[idx])[0][0].item()

                if name_packages_Q[idx] != "":
                    score_package = util.cos_sim(_name_package_from_user_Q_encode, package_encodes_case1[idx])[0][
                        0].item()
                else:
                    score_package = 0.0

                average_score = score_question * ratio_question_package + score_package * (1 - ratio_question_package)
                score_answer.append((average_score, A_part))

            # compute score for case1:
            for idx, i in enumerate(indexes_case2):
                score_context = util.cos_sim(question_user_encode, contexts_case2_encodes[idx])[0][0].item()

                if name_packages_case2[idx] != "":
                    score_package = util.cos_sim(_name_package_from_user_Q_encode, package_encodes_case2[idx])[0][
                        0].item()
                else:
                    score_package = 0.0

                average_score = score_context * ratio_question_package + score_package * (1 - ratio_question_package)
                score_answer.append((average_score, contexts_case2[idx]))
            # v2 */

            score_answer.sort(reverse=True)
            context_string = "".join(score_answer[0][1])
            # four_the_best = score_answer[0:4]
            array_del = []
            # for i in range(len(four_the_best)-1):
            #     if four_the_best[i][0]-four_the_best[i+1][0]<=0.01 and four_the_best[i][0]-four_the_best[i+1][0]>=0:
            #         array_del.append(i+1)
            # for i in reversed(array_del):
            #     del four_the_best[i]

            # print(four_the_best)
            # four_the_best=four_the_best[0]
            # context_string = "".join(four_the_best[0][1])
            # context_string = "\n\n".join([doc for _, doc in four_the_best])
            return context_string, 1    # 1 is flag
            # print(context_string)

        else:
            return question_user, 0     # 0 is flag

    def ask(self, question: str):  # ask normal
        """if flag at _get_context_and_clean == 1 this is a question about package
        is flag at _get_context_and_clean ==0 this is a question about knowledge pretrained of model"""

        context, flag = self._get_context_and_clean(question_user=question, k_search_question=10, k_search_package=5,
                                                    ratio_question_package=0.4)
        qa_chain = self.QA_chain(flag, context, question)

        if flag == 1:
            response = qa_chain.invoke({
                "context": context,
                "question": question
            })
        else:
            response = qa_chain.invoke({
                "question": question
            })
        return response

    def ask_stream(self, question: str):
        """model answer with mode Streaming"""

        context, flag = self._get_context_and_clean(question_user=question, k_search_question=10, k_search_package=5,
                                                    ratio_question_package=0.4)

        prompt_obj = self.create_prompt(flag, context, question)
        if flag == 1:
            prompt = prompt_obj.template.format(context=context, question=question)
        else:
            prompt = prompt_obj.template.format(question=question)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=256,
            temperature=0.4,
            streamer=streamer
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        print("assistant:", end=" ", flush=True)

        def count_real_words(text):
            clean_text = re.sub(r"\s+", " ", text).strip()
            words = clean_text.split()
            return len([w for w in words if w.isalnum()])

        seen_chunks = []
        full_response = ""
        min_response_words = 20

        def last_two_sentences_equal(text):
            sentences = re.split(r'[.\n]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) >= 2:
                return sentences[-1] == sentences[-2]
            return False

        for new_text in streamer:
            print(new_text, end="", flush=True)
            seen_chunks.append(new_text.strip())
            full_response += new_text

            # Case 1: Nếu xuất hiện 4 dấu xuống dòng liên tiếp
            if "\n\n\n\n" in full_response:
                # print("\n[Dừng vì có 4 dòng trống liên tiếp]")
                break

            # Case 2: Nếu 2 câu liên tiếp giống nhau
            if count_real_words(full_response) > min_response_words:
                if last_two_sentences_equal(full_response):
                    # print("\n[Dừng vì lặp câu liên tiếp]")
                    break

def main():
    rag_pipeline = RAG_answer_package_viettel(name_model_llm="vilm/vinallama-2.7b-chat",
                                              name_model_embedding="keepitreal/vietnamese-sbert",
                                              data_path_for_RAG=["/kaggle/input/new-data-vtdb/train",
                                                                 "/kaggle/input/new-data-vtdb/processed"],
                                              vector_store_path="/kaggle/input/new-data-vtdb/vectorstore_qa",
                                              # folder path vector store saved by FAISS
                                              llm=None)

    rag_pipeline.ask_stream("ưu đãi gói cước ST70K?")
    rag_pipeline.ask_stream("Tôi muốn biết thông tin về gói cước NETVT1T_DANGCAP?")

if __name__ == "__main__":
    main()