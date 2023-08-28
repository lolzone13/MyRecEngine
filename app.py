import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    
    return df

@st.cache_data
def load_article_data(filepath):
    df = pd.read_csv(filepath)
    return df

@st.cache_data
def load_file_data():
    f = open("data/productembed.txt", "r")
    text = f.read()
    f.close()

    return eval(text)


df_rl = load_data('data/final_data.csv')
df_art = load_article_data('data/articles.csv')
categories = ['Underwear', 'Garment Upper body', 'Swimwear',
              'Garment Lower body', 'Nightwear', 'Shoes', 'Accessories',
              'Garment Full body', 'Socks & Tights', 'Unknown',
              'Underwear/nightwear']

st.title("Personalised Product Recommendations")
st.divider()
def parse_sequence_string(s):
    n = len(s)
    s = s[1:n-1]
    final_list = []
    f = False
    temp = ""
    for i in s:
        if (i == "'"):
            if not f:
                temp = ""
                f = True
            else:
                final_list.append(temp)
                f = False
        elif i == ' ':
            if f:
                temp += i
        else:
            temp += i
    return final_list


class ThompsonSampling:
    def __init__(self, categories):
        self.num_arms = len(categories)
        self.categories = categories
        self.category_map = {categories[i]: i for i in range(len(categories))}
        self.num_pulls = np.zeros(self.num_arms)
        self.num_successes = np.zeros(self.num_arms)

    def clear_variables(self):
        self.num_pulls = np.zeros(self.num_arms)
        self.num_successes = np.zeros(self.num_arms)

    def simulate(self, users_choices):
        num_rounds = len(users_choices)
        for i in range(num_rounds):
            sampled_probabilities = np.random.beta(
                self.num_successes + 1, self.num_pulls - self.num_successes + 1)
            chosen_arm = np.argmax(sampled_probabilities)

            actual_arm = self.category_map[users_choices[i]]
            reward = 0  # could we work on getting
            if (actual_arm == chosen_arm):
                reward += 1

            self.num_pulls[chosen_arm] += 1
            self.num_successes[chosen_arm] += reward

        sampled_probabilities = np.random.beta(
            self.num_successes + 1, self.num_pulls - self.num_successes + 1)
        choose_next_arm = np.argmax(sampled_probabilities)

        return self.categories[choose_next_arm]

    def get_probabilities(self):
        return np.random.beta(self.num_successes + 1, self.num_pulls - self.num_successes + 1)




@st.cache_data
def category_recommendations(user_id, user_history=[], session_count=0):


    ind = df_rl[df_rl['customer_id'] == user_id].iloc[0]
    users_choices_string = ind['category_sequence']
    
    users_choices = parse_sequence_string(users_choices_string)
    if session_count == 0:
    # All users
        for i in df_rl['category_sequence']:
            users_choices = parse_sequence_string(i)
            st.session_state['TS_all'].simulate(users_choices)
        all_user_probabilities = st.session_state['TS_all'].get_probabilities()
        sorted_categories_all = np.array(categories)[np.argsort(all_user_probabilities)][::-1]

        # One user    
        users_choices = parse_sequence_string(df_rl['category_sequence'][0])
        st.session_state['TS_user'].simulate(users_choices)
        user_probabilities = st.session_state['TS_user'].get_probabilities()
        sorted_categories_user = np.array(categories)[np.argsort(user_probabilities)][::-1]


    if len(user_history) > 0:
        st.session_state['TS_user'].simulate(user_history)
        st.session_state['TS_all'].simulate(user_history)

    # ensemble the probabilities
    factor = np.exp(-session_count)
    probs = factor * st.session_state['TS_all'].get_probabilities() + (1 - factor) * \
        st.session_state['TS_user'].get_probabilities()
    sorted_categories_ensemble = np.array(categories)[np.argsort(probs)][::-1]

    return sorted_categories_ensemble

####################################################################################################################


unique_category = df_rl['category'].unique()
cat_data = df_rl.groupby(['category'])
mapping = {}
for item in unique_category:
    data = cat_data.get_group(item)
    products = data['article_id'].unique()
    mapping[item] = products
    
    
purchase_counts = df_rl.groupby(['customer_id', 'article_id']).size().unstack(fill_value=0)
@st.cache_data
def recommend_product(category_seq, n, user_id, purchase_counts, mapping, articles_data):
    
    
    user_idx = list(purchase_counts.index).index(user_id)
    products = {}
    for cat in category_seq:
        
        product_to_keep = mapping[cat]
        category_purchase_count = purchase_counts.loc[:, product_to_keep]
        sparse_category_purchase_counts = sparse.csr_matrix(category_purchase_count)
        cosine_similarities = cosine_similarity(sparse_category_purchase_counts.T)
        user_history = sparse_category_purchase_counts[user_idx].toarray().flatten()

        # Compute the average cosine similarity between the user's purchased items and all other items
        similarities = cosine_similarities.dot(user_history)

        # Get the indices of the user's purchased items
        purchased_indices = np.where(user_history > 0)[0]

        # Set the similarity scores for purchased items to 0
        similarities[purchased_indices] = 0

        # Sort the items by similarity score and return the top n items
        recommended_indices = np.argsort(similarities)[::-1][:n]
        recommended_items = list(category_purchase_count.columns[recommended_indices])
    
        # Remove the items that the user has already purchased
        purchased_items = list(category_purchase_count.columns[category_purchase_count.loc[user_id] > 0])
        recommended_items = [item for item in recommended_items if item not in purchased_items]
        
        rec_items_name=[]
        for items in recommended_items:
            desc = articles_data.prod_name[articles_data['article_id'] == items].iloc[0]
            rec_items_name.append(desc)
        
        products[cat] = (recommended_items, rec_items_name)
    
    return products




@st.cache_data
def CosineSimilarity(A, B):
    cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cosine

@st.cache_data
def product_grouping():
    product_type_grouping = df_rl.groupby(['product_type'])
    return product_type_grouping

ProductMapping = load_file_data()

product_type_grouping = product_grouping()

@st.cache_data
def SimilarProductRecommendation(article_id, n):
    
    article = str(article_id)
    embedding = ProductMapping[article]
    prod_type = df_rl.product_type[df_rl['article_id']==article_id].iloc[0]
    data = (product_type_grouping.get_group(prod_type)).reset_index(drop = True)
    print(data.shape[0])
    similarity_score = []
    for i in range(data.shape[0]):
        articles = data['article_id'][i]
        score = CosineSimilarity(ProductMapping[str(articles)], embedding)
        similarity_score.append((score, articles))
    sorted_score = sorted(similarity_score, key=lambda x: x[0], reverse=True)
    top_n_articles = [score_tuple[1] for score_tuple in sorted_score[1:n]]
    
    product_name = []
    for items in top_n_articles:
            name = df_rl.prod_name[df_rl['article_id'] == items].iloc[0]
            product_name.append(name)
    
    return product_name
    

##################################################################################################################

def visitation_score(posA, posB, ses_len):
    score = (1/np.log10(np.abs(posA - posB) + 2)) * (1/np.log10(ses_len + 2))
    return score

@st.cache_data
def generate_covisitation_matrix(session, n=0):
    co_visit_mat = dict()
    for i in range(len(session)):
        history = eval(session[i].replace("\n", "").replace(" ", ","))[-1*n:]
        history = list(reversed(history))
        ses_len = len(history)
        for j in range(ses_len):
            itemA = history[j]
            for k in range(j+1, ses_len):
                itemB = history[k]
                pair = str(sorted([itemA, itemB]))
                score = visitation_score(j,k,ses_len)
                if co_visit_mat.get(pair) == None:
                    co_visit_mat[pair] = score
                else:
                    co_visit_mat[pair] += score
                    
    return co_visit_mat


def get_top_per_product(co_visit_mat, product_id, n = 10):
    top = []
    for key in co_visit_mat:
        if str(product_id) in key:
            score = co_visit_mat[key]
            key = eval(key)
            key.remove(product_id)
            pair = key[0]
            top.append((pair, score))
            
    top = sorted(top, reverse = True, key = lambda x: x[1])
    top_ids = [x[0] for x in top[:n]]
    return top_ids


def get_names_from_ids(product, id_col, title_col, cat_col, top_ids):
    names = []
    cats = []
    out = []
    for idx in top_ids:
        names.append(product[product[id_col] == idx][title_col].values[0])
        cats.append(product[product[id_col] == idx][cat_col].values[0])
    for i in range(len(names)):
        out.append(names[i] + " | " + cats[i])
    return out

session = list(df_rl['Purchase History'].unique())
covisit_mat = generate_covisitation_matrix(session)
##################################################################################################################
# MAIN

if 'TS_all' not in st.session_state:
    st.session_state['TS_all'] = ThompsonSampling(categories)
if 'TS_user' not in st.session_state:
    st.session_state['TS_user'] = ThompsonSampling(categories)

# test_list = np.random.choice(['Swimwear', 'Shoes', 'Socks & Tights'], size=(1000,))
recs = category_recommendations(
    '000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318', [], 0)

# product recommendation
products = recommend_product(recs, 15, '000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318', purchase_counts, mapping, df_art)


if 'recs' not in st.session_state:
    st.session_state['recs'] = recs
if 'session_count' not in st.session_state:
    st.session_state['session_count'] = 1
if 'products' not in st.session_state:
    st.session_state['products'] = products

choice = st.radio("Pick a view", ("recommendations", "dataframe"))

if choice == "recommendations":
    choose_category = st.radio("choose category: ", st.session_state['recs'])

    if 'user_history' not in st.session_state:
        st.session_state['user_history'] = []
    
    st.session_state['user_history'].append(choose_category)

    print(choose_category)
    st.caption("Picked Category: " + choose_category)
    st.subheader("Recommended Products in this category: ")

    print(len(st.session_state['user_history']))

    if len(st.session_state['user_history']) >= 10:
        st.session_state['session_count']+=1
        st.session_state['recs'] = category_recommendations('000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318', st.session_state['user_history'], st.session_state['session_count'])
        st.session_state['user_history'] = []
    
    products_cat = st.session_state['products'][choose_category][1]
    product_choice = st.radio("choose product: ", list(set(products_cat)))
    product_id = ""
    for i in range(len(products_cat)):
        if products_cat[i] == product_choice:
            product_id = st.session_state['products'][choose_category][0][i]
            break
    
    st.subheader("Similar Products: ")
    similar_products = set(SimilarProductRecommendation(product_id, 20))
    st.write(list(similar_products))

    combo = get_top_per_product(covisit_mat, product_id, 10)

    combo_ids = get_names_from_ids(df_rl ,'article_id','prod_name', 'product_type', combo)

    st.subheader("People who bought this also bought: ")
    st.write(combo_ids)

else:
    st.dataframe(df_rl)



