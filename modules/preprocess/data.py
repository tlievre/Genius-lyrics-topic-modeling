import math
from collections import Counter

def sample_data(df, balanced = True, prop = 0.1):
    if balanced:
        # compute the sorted decreasing parties frequencies
        decade_frequencies = Counter(df['decade']).most_common()
        print(decade_frequencies)

        # retrieve the under represented class
        nb_under_class = decade_frequencies[-1][1]

        # Return a random sample of items from each party following the under sampled number of class
        sample_df = df.groupby("decade").sample(n = nb_under_class, random_state = 500)
    else:
        # create sample df 1/3 of the actual loaded data
        sample_df = df.sample(frac = prop)
    return sample_df


def filter_data(df):

    # Retrieve only the texts identified as English language by both cld3 and fasttext langid
    df = df[df.language == 'en']
    
    # Delete rows containing NaN values
    df.dropna(inplace=True)
    if df[df.isnull().any(axis=1)]:
        raise Exception("Data contain nan value")
    
    # filter values by date (1960 - 2022)
    df = df[(df.year >= 1960) & (df.year < 2023)]
    
    # add decade column
    df['decade'] = df['year'].map(lambda x : int(math.trunc(x / 10) * 10))


