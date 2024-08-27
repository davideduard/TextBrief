import random
import os
import re

from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from gensim.models import KeyedVectors
import nltk
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import math
from statistics import median

app = Flask(__name__)
CORS(app)

os.environ['SSL_CERT_FILE'] = 'cacert.pem'
os.environ['NLTK_DATA'] = '/Users/eduarddavid/nltk_data'

nlp = spacy.load('ro_core_news_lg')
stop_words = set(stopwords.words('romanian'))


# Load word vectors
embedding_file = 'embeddings/corola.300.20.vec'
try:
    word_vectors = KeyedVectors.load("saved_models/word2vec_embeddings.model")
except FileNotFoundError:
    word_vectors = KeyedVectors.load_word2vec_format(embedding_file)
    word_vectors.save("saved_models/word2vec_embeddings.model")


def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    lemmatized_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()

        # remove special characters, punctuation and numbers
        sentence = re.sub(r'\d+', '', sentence)
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # tokenize the sentence and remove stop words
        word_tokens = word_tokenize(sentence)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        clean_text = ' '.join(filtered_text)

        # lemmatize the sentence
        doc = nlp(clean_text)
        lemmatized_text = [token.lemma_ for token in doc]
        lemmatized_sentences.append(lemmatized_text)

    return sentences, lemmatized_sentences


def embed_sentences(lemmatized_sentences):
    # Transform each word into it's embedding
    preprocessed_sentences = []
    for lemmatized_text in lemmatized_sentences:
        embedded_text = []
        for word in lemmatized_text:
            try:
                word_vector = word_vectors[word]
                embedded_text.append(word_vector)
            except KeyError:
                pass
        preprocessed_sentences.append(embedded_text)
    return preprocessed_sentences


def average_score(preprocessed_sentence):
    if len(preprocessed_sentence) == 0:
        return None
    return np.mean(preprocessed_sentence, axis=0)


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def compute_similarity_matrices(sentences, lemmatized_sentences):
    # used for word embedding representation
    preprocessed_sentences = embed_sentences(lemmatized_sentences)
    sentences_embeddings_avg = [average_score(sentence) for sentence in preprocessed_sentences]
    cosine_similarity_matrix = cosine_similarity(sentences_embeddings_avg)

    # used for BOW representation
    sentence_sets = [set(sentence) for sentence in lemmatized_sentences]
    jaccard_similarity_matrix = [[jaccard_similarity(set1, set2) for set2 in sentence_sets] for set1 in sentence_sets]

    lemmatized_sentences_string = [' '.join(sentence) for sentence in lemmatized_sentences]

    # each sentence is represented as a vector of TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(lemmatized_sentences_string)
    cosine_similarity_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_similarity_matrix, jaccard_similarity_matrix, cosine_similarity_tfidf


def visualize_graph_with_scores(graph, scores):
    pos = nx.spring_layout(graph, k=3)  # Layout for good separation
    plt.figure(figsize=(10, 10))

    # Draw the graph
    nx.draw(graph, pos, with_labels=False, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
            edge_color='gray')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    formatted_edge_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=formatted_edge_labels)

    # Draw node labels
    node_labels = {node: str(node + 1) for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)

    median_y = median([y for x, y in pos.values()])

    for node, score in scores.items():
        offset_y = 0.07 if pos[node][1] > median_y else -0.07
        label_pos = (pos[node][0], pos[node][1] + offset_y)
        plt.text(label_pos[0], label_pos[1], f"{score:.3f}", fontsize=10, color='red', ha='center', va='center')

    plt.title('Sentence Similarity Graph with PageRank Scores')
    plt.show()


def text_rank(sentences, similarity_matrix, compression_rate=0.3):
    graph = nx.Graph()
    number_of_sentences = len(sentences)
    for sentence in range(number_of_sentences):
        graph.add_node(sentence)
    for i in range(number_of_sentences):
        for j in range(i + 1, number_of_sentences):
            similarity_score = similarity_matrix[i][j]
            graph.add_edge(i, j, weight=similarity_score)

    scores = page_rank(graph)

    # visualize_graph_with_scores(graph, scores)

    ranked_sentences = [(scores[i], i, sentence) for i, sentence in enumerate(sentences)]
    ranked_sentences.sort(key=lambda x: x[0], reverse=True)

    top_n = math.ceil(compression_rate * len(sentences))
    top_sentences = [sentence for _, _, sentence in ranked_sentences[:top_n]]
    top_sentences.sort(key=lambda x: sentences.index(x))

    return top_sentences


def page_rank(graph, damping_factor=0.85, tol=1e-6):
    n = len(graph.nodes())
    # Initialize pagerank scores
    pagerank_scores = {node: random.random() for node in graph.nodes}

    while True:
        new_pagerank_scores = {node: (1 - damping_factor) for node in graph.nodes}
        # Accumulate scores from neighbors
        for node in graph.nodes:
            total_weight = sum(graph[node][n]['weight'] for n in graph.neighbors(node))
            if total_weight == 0:
                continue
            for neighbor in graph.neighbors(node):
                weight = graph[node][neighbor]['weight']
                new_pagerank_scores[neighbor] += damping_factor * pagerank_scores[node] * weight / total_weight

        # Check for convergence
        if max(abs(new_pagerank_scores[node] - pagerank_scores[node]) for node in graph.nodes) < tol:
            break

        pagerank_scores = new_pagerank_scores

    return pagerank_scores

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')

    #text = 'Potrivit unui comunicat de presă al fundației ECDL în 2008 aproximativ 83 % din populația globală nu folosea Internetul. În iulie 2016, o rezoluție ONU a declarat că accesul la internet este un drept uman de bază. Potrivit unui studiu întocmit de firma de cercetare on-line InternetWorldStats, în noiembrie 2007 rata de penetrare a Internetului în România a atins nivelul de 31,4 % din totalul populației, estimată la 22,27 milioane de locuitori, iar numărul de conexiuni broadband era de 1.769.300. Între 2007 - 2011 numărul conexiunilor la Internet în gospodăriile românești a crescut de la 22 % la 47 %; cifrele corespunzătoare la nivelul Uniunii Europene au fost 54 % și 73 %. În ziua de astăzi Internetul este susținut și întreținut de o mulțime de firme comerciale. Protocoalele fundamentale ale Internetului, care asigură interoperabilitatea între orice două calculatoare sau aparate inteligente care le implementează, sunt Internet Protocol (IP), Transmission Control Protocol (TCP) și User Datagram Protocol (UDP). Aceste trei protocoale reprezintă însă doar o parte din nivelul de bază al sistemului de protocoale Internet, care mai include și protocoale de control și aplicative, cum ar fi: DNS, PPP, SLIP, ICMP, POP3, IMAP, SMTP, HTTP, HTTPS, SSH, Telnet, FTP, LDAP, SSL, WAP și SIP. Din cauza multelor fuziuni dintre companiile de telefonie și cele de Internet (Internet Service Providers, prescurtat ISP) au apărut o serie de probleme în sensul că sarcinile acestora nu erau clar delimitate. Rețeaua regională a ISP-ului este formată prin interconectarea ruterelor din diverse orașe pe care le deservește compania. Dacă pachetul este destinat unui calculator-gazdă deservit direct de către rețeaua ISP, pachetul va fi livrat direct lui. În partea superioară a acestei ierarhii se găsesc operatorii principali de la nivelul backbone-ului rețelei, companii cum ar fi AT&T sau SPRINT. Corporațiile și firmele de hosting utilizează așa-numitele „ferme” de servere rapide (= multe servere, situate eventual în aceeași sală sau clădire), conectate direct la backbone. Operatorii încurajează pe clienții lor să folosească această conectare directă prin închirierea de spațiu în rack-uri = dulapuri speciale standardizate pentru echipamentul clientului, care se află în aceeași cameră cu ruterul, conducând la conexiuni scurte și rapide între fermele de servere și backbone-ul rețelei. Dacă un pachet trimis în backbone este destinat unui ISP sau unei companii deservite de aceeași coloană, el este transmis celui mai apropiat ruter. Pentru a permite pachetelor să treacă dintr-un backbone în altul, acestea sunt conectate în NAP-uri (Network Access Point). O rețea locală conectează toate aceste rutere astfel încât pachetele să poată fi retransmise rapid din orice coloană în orice alta. Unul dintre paradoxurile Internetului este acela că ISP-urile, care se află în competiție între ele pentru câștigarea de clienți, cooperează în realizarea de conectări private și întreținerea Internetului. Există un șir întreg de metode de cuplare fizică a unui calculator sau aparat „inteligent” (smart) la Internet. Accesul unui utilizator la Internet prin intermediul rețelei de telefon analogice fixe tradiționale: utilizatorul unui calculator cheamă programul de comunicație necesar, care mai întâi se conectează la modem. Modemul este o componentă a calculatorului care convertește semnalele digitale (de transmis) în semnale analogice, care pot circula în rețeaua telefonică. Semnalele modulate (de fapt datele) sunt transferate la punctul de livrare (Point Of Presence, POP) al ISP-ului, unde sunt preluate din sistemul telefonic și transferate în rețeaua regională de Internet a ISP-ului. Din acest punct sistemul este în întregime digital și se bazează pe comutarea de pachete (packet switching); în acest sistem de transmisie informația care trebuie transmisă este "mărunțită" în multe pachete mici, care sunt apoi transmise la destinație în mod independent unele de altele și chiar pe căi diferite; sigur că la destinație pachetele trebuie reasamblate în ordinea corectă. Pe lângă utilizarea rețelei fixe publice aceeași tehnică se poate folosi și pe linii fixe dedicate (închiriate). Acest tip de acces a rămas în urmă ca viteză și siguranță în funcționare și nu se mai utilizează aproape deloc. Legătură prin radio, de la un telefon celular de tip smartphone, de la un calculator portabil sau, mai general, de la un dispozitiv Internet mobil la antena celulară terestră, utilizând tehnicile GSM sau UMTS. Această tehnologie a fost dezvoltată pentru a asigura accesul în zonele izolate la internet de bandă largă, sau în zone în care furnizorii clasici nu au infrastructura dezvoltată. Punctul de pornire în dezvoltarea Internetului a fost rivalitatea între cele două mari puteri ale secolului al XX-lea: Statele Unite ale Americii și Uniunea Sovietică. Acest fapt a declanșat o îngrijorare deosebită în Statele Unite ale Americii, astfel președintele Eisenhower înființează o agenție specială subordonată Pentagonului: Advanced Research Projects Agency. În 1959 John McCarthy, profesor la Universitatea Stanford, al cărui nume va fi asociat cu inteligența artificială, găsește soluția de a conecta mai multe terminale la un singur calculator central: time-sharing (partajarea timpului). Aceasta este o modalitate de lucru în care mai multe aplicații (programe de calculator) solicită acces concurențial la o resursă (fizică sau logică), prin care fiecărei aplicații i se alocă un anumit timp pentru folosirea resursei solicitate. Apărând apoi primele calculatoare în marile universități se pune problema interconectării acestora. Astfel, pentru a transmite informația, aceasta este mărunțită în porțiuni mici, denumite pachete. Ca și la poșta clasică, fiecare pachet conține informații referitoare la destinatar, astfel încât el să poată fi corect dirijat pe rețea. La destinație întreaga informație este reasamblată. Deși această metodă întâmpină rezistență din partea specialiștilor, în 1969 începe să funcționeze rețeaua "ARPANET" între 4 noduri: University of California din Los Angeles (UCLA), University of California din Santa Ana, University of Utah și Stanford Research Institute (SRI). Toate acestea au fost codificate într-un protocol care reglementa transmisia de date. În forma sa finală, acesta era TCP/IP (Transmission Control Protocol / Internet Protocol), creat de Vint Cerf și Robert Kahn în 1970 și care este și acum baza Internetului.'
    # It does not work with romanian language
    # text_with_punctuation = restore_punctuation(text)
    # print(text_with_punctuation)

    # sentences = segment_sentences(text_with_punctuation)

    sentences, lemmatized_sentences = preprocess_text(text)
    cosine_sim_matrix, jaccard_sim_matrix, cosine_sim_tfidf = compute_similarity_matrices(sentences, lemmatized_sentences)
    #print("embd:")
    top_sentences_cosine = text_rank(sentences, cosine_sim_matrix)
    #print("tf-idf:")
    top_sentences_tfidf = text_rank(sentences, cosine_sim_tfidf)
    #print("bow:")
    top_sentences_jaccard = text_rank(sentences, jaccard_sim_matrix)

    return jsonify({
        'embeddings': ' '.join(top_sentences_cosine),
        'jaccard': ' '.join(top_sentences_jaccard),
        'tfidf': ' '.join(top_sentences_tfidf)
    })


if __name__ == '__main__':
    #summarize()
    app.run(debug=True)
