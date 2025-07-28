#huggingface_hub[hf_xet]
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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAG_vi_mrc_large:
    def __init__(self,
                 name_model_llm:str = "nguyenvulebinh/vi-mrc-large",
                 name_model_embedding :str = "keepitreal/vietnamese-sbert",
                 name_model_find_relevant : str = None,
                 data_path :str = None,
                 vectors_store_path :str = None,
                 llm : Any = None):

        self.name_model_llm = name_model_llm
        self.name_model_embedding = name_model_embedding
        self.data_path = data_path
        self.vectors_store_path = vectors_store_path
        self.llm=llm

        self.model ,self.tokenizer = self.model_tokenizer()
        self.model_embedding = HuggingFaceEmbeddings(
                                                    # model_name=self.name_model_embedding,
                                                    cache_folder=self.name_model_embedding,
                                                    model_kwargs={"device": device},
                                                    show_progress=True
                                                    )

        self.model_find_relevant_context = SentenceTransformer(name_model_find_relevant)

        self.vector_store = self.load_vectors_store()

    def build_vectors_store(self):

        def case_for_folder_one(folder_path):
            case1=[]
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

                    case1.append(a_conversation)
            return case1

        def case_for_folder_two(folder_path):
            case2=[]
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
                    case2.append(a_conten)
            return case2

        if isinstance(self.data_path , str):
            case1=case_for_folder_one(self.data_path )

            documents = [Document(page_content=text) for text in case1]

            vector_db = FAISS.from_documents(documents, self.model_embedding)
            vector_db.save_local(self.vectors_store_path)

        elif isinstance(self.data_path , list):
            case1=case_for_folder_one(self.data_path[0])
            case2=case_for_folder_two(self.data_path[1])

            text_from_json=case1+case2
            documents = [Document(page_content=text) for text in text_from_json]

            vector_db = FAISS.from_documents(documents, self.model_embedding)
            vector_db.save_local(self.vectors_store_path)

    def load_vectors_store(self):
        if not os.path.exists(self.vectors_store_path):
            print("chưa có vectors store")
            self.build_vectors_store()
            vectors_store = FAISS.load_local( folder_path = self.vectors_store_path,
                                              embeddings = self.model_embedding,
                                              allow_dangerous_deserialization = True
                                              )
            return vectors_store
        else:

            vectors_store = FAISS.load_local(folder_path=self.vectors_store_path,
                                             embeddings=self.model_embedding,
                                             allow_dangerous_deserialization=True
                                             )
            return vectors_store
    def model_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.name_model_llm, local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(self.name_model_llm ,local_files_only=True)
        return model, tokenizer

    def truncate_context(self, question, context, max_len=512):
        q_ids = self.tokenizer.encode(question, add_special_tokens=False)
        c_ids = self.tokenizer.encode(context, add_special_tokens=False)

        total_len = len(q_ids) + len(c_ids) + 3  # [CLS], [SEP], [SEP]

        if total_len <= max_len:
            return context

        c_max_len = max_len - len(q_ids) - 3
        c_ids = c_ids[:c_max_len]
        truncated_context = self.tokenizer.decode(c_ids, skip_special_tokens=True)
        return truncated_context

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
            # context_string = "".join(score_answer[0][1])
            four_the_best = score_answer[0:2]
            # array_del = []
            # for i in range(len(four_the_best)-1):
            #     if four_the_best[i][0]-four_the_best[i+1][0]<=0.001 and four_the_best[i][0]-four_the_best[i+1][0]>=0:
            #         array_del.append(i+1)
            # for i in reversed(array_del):
            #     del four_the_best[i]

            # print(four_the_best)
            # four_the_best=four_the_best[0]
            # context_string = "".join(four_the_best[0][1])
            context_string = ".\n".join([doc for _, doc in four_the_best])
            return context_string, 1  # 1 is flag
            # print(context_string)

        else:
            return question_user, 0  # 0 is flag
    def QA_chain(self):
        return pipeline("question-answering", model=self.model, tokenizer=self.tokenizer, device_map=0)

    def ask(self, question: str):  # ask normal

        context, flag= self._get_context_and_clean(question_user=question, k_search_question=20, k_search_package=10,
                                                   ratio_question_package=0.5)
        if flag == 1:
            qa_chain = self.QA_chain()
            context = self.truncate_context(question=question, context=context)
            response = qa_chain(question=question, context=context)
        else:
            response = {"answer":"tôi chưa được cấu hình để trả lời câu hỏi của bạn. Tôi chỉ có thể trả lời thông tin liên quan đến gói cước của nhà mạng Viettel."}
        return response
