from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def save_wordcloud(data, bgcolor, title):
    plt.figure(figsize = (50, 50))
    wc = WordCloud(background_color=bgcolor, max_words=1000, max_font_size=50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
    filepath = title + '.png'
    plt.savefig(filepath)


my_filepath="IMDB Dataset.csv"
df = pd.read_csv(my_filepath)
pos_rev = df[df.sentiment == 'positive']
neg_rev = df[df.sentiment == 'negative']
save_wordcloud(pos_rev.review, 'black', 'pos')
save_wordcloud(neg_rev.review, 'black', 'neg')

