import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

N = 100  # Genres
p = 0.8  # Percentage for training/testing split

# Read the original data
data = pd.read_csv('data_w_genres.csv')

# Delete lines without genres
data = data[data['genres'].apply(lambda x: len(eval(x)) > 0)]

# Expand genres column
data_expanded = data.assign(genres=data['genres'].apply(eval)).explode('genres')

# Keep only the lines with genres in the top N most represented
genre_counts = data_expanded['genres'].value_counts()
top_N_genres = genre_counts.head(N).index
data_filtered = data_expanded[data_expanded['genres'].isin(top_N_genres)]

# Save filtered data
output_folder = 'Output'
os.makedirs(output_folder, exist_ok=True)
data_filtered.to_csv(f'{output_folder}/filtered_data.csv', index=False)

# Encode artists if not already encoded
artist_label_encoder = LabelEncoder()
data_filtered = data_filtered.assign(Artists_encoded=artist_label_encoder.fit_transform(data_filtered['artists']))
data_filtered = data_filtered.drop(columns=['artists'])

# Save encoded artists data
mapping_folder = 'Mapping'
os.makedirs(mapping_folder, exist_ok=True)
data_filtered.to_csv(f'{output_folder}/Artists_encoded.csv', index=False)

# Save the mapping between artists and encoded values
mapping_data_artists = pd.DataFrame({'artists': artist_label_encoder.classes_, 'Artists_encoded': range(len(artist_label_encoder.classes_))})
mapping_data_artists.to_csv(f'{mapping_folder}/artists_mapping.csv', index=False)





# Generate datasets with and without encoding for genres and inclusion of artists
for encode_genres in [True, False]:
    for include_artists in [True, False]:
        df = data_filtered.copy()

        if not include_artists:
            df = df.drop(columns=['Artists_encoded'])

        if encode_genres:
            # Encode genres if not already encoded
            label_encoder = LabelEncoder()
            df = df.assign(genres_encoded=label_encoder.fit_transform(df['genres']))
            df = df.drop(columns=['genres'])

            mapping_genres = pd.DataFrame({'genres':        label_encoder.classes_,'genres_encoded':range(len(label_encoder.classes_))})
            mapping_genres.to_csv(f'{mapping_folder}/genres_mapping.csv',index=False)



        # Generate datasets with p% of each genre going to the training set
        train_df_genre, test_df_genre = train_test_split(df, test_size=1-p, stratify=df['genres_encoded'] if encode_genres else df['genres'], random_state=42)

        # Save the datasets for p% of each genre
        folder_name_genre = f'{output_folder}/{"Encoded" if encode_genres else "NotEncoded"}_Genres{"_ArtistsEncoded" if include_artists else "_ArtistsNotEncoded"}/p%ofgenre'
        os.makedirs(folder_name_genre, exist_ok=True)
        train_df_genre.to_csv(f'{folder_name_genre}/Train.csv', index=False)
        test_df_genre.to_csv(f'{folder_name_genre}/Test.csv', index=False)

        # Generate datasets with p% of the original dataset going to the training set
        train_df_random, test_df_random = train_test_split(df, test_size=1-p, random_state=42)

        # Save the datasets for p% of the original dataset
        folder_name_random = f'{output_folder}/{"Encoded" if encode_genres else "NotEncoded"}_Genres{"_ArtistsEncoded" if include_artists else "_ArtistsNotEncoded"}/p%random'
        os.makedirs(folder_name_random, exist_ok=True)
        train_df_random.to_csv(f'{folder_name_random}/Train.csv', index=False)
        test_df_random.to_csv(f'{folder_name_random}/Test.csv', index=False)
