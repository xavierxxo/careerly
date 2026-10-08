from flask import Flask, render_template, request
import pickle
import re
import mysql.connector
import os

from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


app = Flask(__name__)

MODEL_PATH = 'model/model_d_tfidf_centroid.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

tfidf = model['vec']
centroid_tfidf = model['centroid_tfidf']
kategori_labels = model['cats']

# Connect ke Database
def get_db():

    config = {
        'host': os.environ.get('DB_HOST', '127.0.0.1'),
        'port': int(os.environ.get('DB_PORT', 3306)),
        'user': os.environ.get('DB_USER', 'root'),
        'password': os.environ.get('DB_PASSWORD', ''),
        'database': os.environ.get('DB_NAME', 'sistem_rekomendasi_karier'),
        'charset': 'utf8mb4',
    }

    # berkas sertifikat CA dari Aiven (ca.pem)
    ca = os.environ.get('DB_SSL_CA')
    if ca:
        config['ssl_ca'] = ca

    return mysql.connector.connect(**config)

# Mengambil informasi kategori dari database MySQL
def get_kategori_info(nama_kategori):

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT *
        FROM kategori_karier
        WHERE nama_kategori = %s
        """,
        (nama_kategori,)
    )

    kategori = cursor.fetchone()

    if kategori:

        id_kat = kategori['id_kategori']

        # Technical skills
        cursor.execute(
            """
            SELECT nama_skill
            FROM skill
            WHERE id_kategori = %s
              AND tipe_skill = 'technical'
            """,
            (id_kat,)
        )

        technical = [
            r['nama_skill']
            for r in cursor.fetchall()
        ]

        # Soft skills
        cursor.execute(
            """
            SELECT nama_skill
            FROM skill
            WHERE id_kategori = %s
              AND tipe_skill = 'soft'
            """,
            (id_kat,)
        )

        soft = [
            r['nama_skill']
            for r in cursor.fetchall()
        ]

        kategori['technical_skills'] = technical
        kategori['soft_skills'] = soft

    cursor.close()
    db.close()

    return kategori

# Preprocessing
STOPWORDS = set("""
a an the and or of to in for with on at by is are be as
we you your our will that this from have has can

job jobs role roles team teams work working company companies
please apply applicant applicants candidate candidates opportunity
benefit benefits salary insurance office jakarta indonesia bandung
surabaya remote hybrid onsite

year years experience minimum required requirement requirements
responsibility responsibilities qualification qualifications
ability strong good excellent well plus about join looking hiring
position full part time day days month months new fresh graduate
graduates ideal preferred

skill skills knowledge
using use used ensure support provide develop development
manage management

least also including related within across
etc based need needed make great high best key part
relevant various detail fast dynamic environment seeking
""".split())


def preprocessing(text):

    text = str(text)

    # lowercase
    text = text.lower()

    # hapus URL
    text = re.sub(
        r'http\S+|www\.\S+',
        ' ',
        text
    )

    # hanya huruf
    text = re.sub(
        r'[^a-z\s]',
        ' ',
        text
    )

    # hapus stopwords dan token pendek
    tokens = [
        token
        for token in text.split()
        if len(token) > 2
        and token not in STOPWORDS
    ]

    return ' '.join(tokens)

#Rekomendasi Model D
def rekomendasikan(
    user_input,
    top_n=7
):
    # Preprocessing profil user
    text_preprocessed = preprocessing(
        user_input
    )

    # Ubah profil menjadi TF-IDF vector
    user_vector = tfidf.transform(
        [text_preprocessed]
    )

    # Cosine Similarity
    scores = cosine_similarity(
        user_vector,
        centroid_tfidf
    )[0]

    # Normalisasi score rekomendasi
    scores_clipped = [
        max(0.0, float(score))
        for score in scores
    ]

    total_score = sum(
        scores_clipped
    )

    # Menyimpan hasil
    hasil = []

    for i, score in enumerate(scores_clipped):

        if total_score > 0:

            similarity = (
                score /
                total_score *
                100
            )

        else:

            similarity = 0.0

        hasil.append({

            'kategori': kategori_labels[i],

            'similarity': round(
                similarity,
                2
            )
        })

    # Ranking
    hasil = sorted(
        hasil,
        key=lambda x: x['similarity'],
        reverse=True
    )

    # Menampilkan 7 kategori
    return hasil[:top_n]

# Routes
@app.route('/')
def index():

    return render_template(
        'index.html'
    )


@app.route(
    '/rekomendasi',
    methods=['POST']
)
def rekomendasi():

    user_input = request.form.get(
        'profil',
        ''
    ).strip()

    major_input = request.form.get(
        'major',
        ''
    ).strip()

    # Validasi panjang deskripsi
    if not user_input or len(user_input) < 500:

        error = (
            "Profile description too short! "
            "Please tell us more about yourself "
            "(minimum 500 characters)."
        )

        return render_template(
            'index.html',
            error=error,
            profil=user_input,
            major=major_input
        )

    # Validasi bahasa Inggris
    try:

        bahasa = detect(
            user_input
        )

        if bahasa != 'en':

            error = (
                "Please use English! Our system detects "
                "your profile text is written in another "
                "language. Kindly provide your description "
                "in English for accurate analysis."
            )

            return render_template(
                'index.html',
                error=error,
                profil=user_input,
                major=major_input
            )

    except Exception:

        error = (
            "System failed to recognize the text language. "
            "Please make sure to use a clear English sentence structure."
        )

        return render_template(
            'index.html',
            error=error,
            profil=user_input,
            major=major_input
        )

    # Proses rekomendasi
    try:

        hasil = rekomendasikan(
            user_input,
            top_n=7
        )

        # mengambil data kategori dari MySQL
        hasil_lengkap = []

        for item in hasil:

            info = get_kategori_info(
                item['kategori']
            )

            hasil_lengkap.append({

                'kategori': item['kategori'],

                'similarity': item['similarity'],

                'deskripsi': (
                    info['deskripsi']
                    if info
                    else
                    'Career path details are currently unavailable.'
                ),

                'icon': (
                    info['icon']
                    if info
                    else
                    'briefcase'
                ),

                'technical_skills': (
                    info['technical_skills']
                    if info
                    else
                    []
                ),

                'soft_skills': (
                    info['soft_skills']
                    if info
                    else
                    []
                ),
            })

        # Menampilkan hasil
        return render_template(
            'result.html',
            hasil=hasil_lengkap,
            profil=user_input
        )
    
    except Exception as e:

        error_msg = (
            f"Internal System Error: {str(e)}"
        )

        return render_template(
            'index.html',
            error=error_msg,
            profil=user_input
        )

# Run
if __name__ == '__main__':

    app.run(
        debug=False
    )    