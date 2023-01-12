from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from dotenv import load_dotenv

from .models import Question

import pandas as pd
import openai
import numpy as np

from resemble import Resemble

import os

load_dotenv('.env')

Resemble.api_key(os.environ["RESEMBLE_API_KEY"])
openai.api_key = os.environ["OPENAI_API_KEY"]

COMPLETIONS_MODEL = "text-davinci-003"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 150,
    "model": COMPLETIONS_MODEL,
}

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> tuple[str, str]:
    """
    Fetch relevant embeddings
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[df['title'] == section_index].iloc[0]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            space_left = MAX_SECTION_LEN - chosen_sections_len - len(SEPARATOR)
            chosen_sections.append(SEPARATOR + document_section.content[:space_left])
            chosen_sections_indexes.append(str(section_index))
            break

        chosen_sections.append(SEPARATOR + document_section.content)
        chosen_sections_indexes.append(str(section_index))

    header = """Please keep your answers to three sentences maximum, and speak in complete sentences.:\n"""

    question_1 = "\n\n\nQ: How do I become a Traveler with Medical Solutions?\n\nA: Simply apply here. You can also call us at 1.866.633.3548 and speak with a recruiter, who can answer your questions and send you an information packet. All we need to begin is your application and resume. Once we receive your information, we can begin discussing potential assignments that fit your profile. When you find a job you want, your recruiter will submit you for the job and walk you through the process from there."
    question_2 = "\n\n\nQ: Am I obligated to travel with Medical Solutions if I fill out an application?\n\nA:No. Your information is always kept confidential and we must get your permission before submitting your resume to a client. This gives you a chance to explore your options and choose the best assignment before signing a contract."
    question_3 = "\n\n\nQ: Is there a fee to become a Traveler for Medical Solutions?\n\nA: Absolutely not. We’re paid by the hospital. If you choose to travel with Medical Solutions, you will become our employee and we pay you."
    question_4 = "\n\n\nQ: Where can I find travel healthcare jobs and what jobs do you offer?\n\nA: Medical Solutions has open travel healthcare assignments nationwide in a variety of specialties and shifts. You can search our extensive database of travel healthcare positions by specialty, title, and location here. You can also sign up for our jobs email which sends you available jobs in your specialty."
    question_5 = "\n\n\nQ: How long are travel healthcare assignments?\n\nA: Most assignments are 13 weeks in length, but we’ve seen them as short as four weeks and as long as 24. You are obligated to finish your assignment as contracted, but there is no contract binding you to work more assignments afterward. You can take a new assignment right after your last or take a break. It’s all up to you! How long does it take to get an assignment? Once you have interviewed for the position and both you and the facility decide to move forward, it typically takes one to six weeks, depending on when you are available and the start date of the assignment. We typically begin presenting new assignments to you about 30 days from the end of your current assignment."
    question_6 = "\n\n\nQ: What about state licensing?\n\nA: Many states are in the Nurse Licensure Compact (NLC), which allows you reciprocal rights to practice in compact states. If you need to obtain a new state license, your Recruiter can tell you exactly how much time is necessary to apply for it and the cost. If a new license is required for your assignment, Medical Solutions will reimburse you for the cost!"
    question_7 = "\n\n\nQ:What happens after my assignment is over?\n\nA: Before your assignment ends, your recruiter will begin discussing new travel opportunities with you. However, it is completely your choice to extend your assignment (if an extension is offered), take another assignment, go full time, or simply take a break. Get a jump on exploring new jobs here."
    question_8 = "\n\n\nQ: What are the benefits of traveling vs a permanent position?\n\nA: Many travelers say the variety of travel jobs helps them avoid nurse burnout, plus it’s a great way to develop new skills and it looks great on your resume! Traveling is also exciting, especially if you’re flexible with your time, family, and social life. You often receive higher pay, paid lodging, preferred scheduling, and bonuses through your company’s loyalty program. Assignments are typically 13 weeks in duration, long enough to enjoy your new city and maybe even make a few friends, but not so long you get bored. This lifestyle is best suited to those who can easily adapt to new surroundings, are flexible with location, and confident in their clinical skills."
    question_9 = "\n\n\nQ: What do I do if I get injured on the job?\n\nA: If you are injured at work and it is an emergency, call 911 or go to the nearest emergency room. If your injury is not an emergency, you must call Conduent at 844-975-3471 prior to seeking treatment. Conduent is a 24/7 injury triage service that provides immediate expertise from a registered nurse (RN) to accurately assess the severity of a workplace injury and recommend the best course of action. After coordinating a referral for medical treatment, Conduent will send a report of injury to us and we will file a workers’ compensation claim with our claim administrator, CCMSI. Our Work Comp team will provide you with your claim information and further instructions via email."
    question_10 = "\n\n\nQ: How much will I be paid?\n\nA: Your total compensation package — including your hourly pay, benefits, bonuses, reimbursements, etc. — is completely customized to fit your needs. Pay rates vary from assignment to assignment depending on location, the hospital, your specialty, and other factors."

    return (header + "".join(chosen_sections) + question_1 + question_2 + question_3 + question_4 + question_5 + question_6 + question_7 + question_8 + question_9 + question_10 + "\n\n\nQ: " + question + "\n\nA: "), ("".join(chosen_sections))

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
) -> tuple[str, str]:
    prompt, context = construct_prompt(
        query,
        document_embeddings,
        df
    )

    print("===\n", prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n"), context

def index(request):
    return render(request, "index.html", { "default_question": "How do I become a Traveler with Medical Solutions?" })

@csrf_exempt
def ask(request):
    question_asked = request.POST.get("question", "")

    if not question_asked.endswith('?'):
        question_asked += '?'

    previous_question = Question.objects.filter(question=question_asked).first()
    audio_src_url = previous_question and previous_question.audio_src_url if previous_question else None

    if audio_src_url:
        print("previously asked and answered: " + previous_question.answer + " ( " + previous_question.audio_src_url + ")")
        previous_question.ask_count = previous_question.ask_count + 1
        previous_question.save()
        return JsonResponse({ "question": previous_question.question, "answer": previous_question.answer, "audio_src_url": audio_src_url, "id": previous_question.pk })

    df = pd.read_csv('Embeddings.pdf.pages.csv')
    document_embeddings = load_embeddings('Embeddings.pdf.embeddings.csv')
    answer, context = answer_query_with_context(question_asked, df, document_embeddings)

    project_uuid = '7ac56bc3'
    voice_uuid = '4e42dcc8'

    response = Resemble.v2.clips.create_sync(
        project_uuid,
        voice_uuid,
        answer,
        title=None,
        sample_rate=None,
        output_format=None,
        precision=None,
        include_timestamps=None,
        is_public=None,
        is_archived=None,
        raw=None
    )


    # question = Question(question=question_asked, answer=answer, context=context, audio_src_url=response['item']['audio_src'])
    question = Question(question=question_asked, answer=answer, context=context)
    question.save()

    return JsonResponse({ "question": question.question, "answer": answer, "audio_src_url": question.audio_src_url, "id": question.pk })

@login_required
def db(request):
    questions = Question.objects.all().order_by('-ask_count')

    return render(request, "db.html", { "questions": questions })

def question(request, id):
    question = Question.objects.get(pk=id)
    return render(request, "index.html", { "default_question": question.question, "answer": question.answer, "audio_src_url": question.audio_src_url })
