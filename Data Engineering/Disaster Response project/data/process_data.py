import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load the messages and categories, merge them as a DataFrame. """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """ Clean the data inside the given DataFrame. """
    categories = pd.Series(df["categories"]).str.split(pat=";", n=36, expand=True)
    first_row = categories.iloc[0].tolist()
    category_colnames = [name[:-2] for name in first_row]
    categories.columns = category_colnames
    # Change 'related-2' to 'related-1'
    categories['related'] = categories['related'].apply(lambda val: 'related-1' if val == 'related-2' else val)
    
    # Convert category values to numbers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # Drop the categories from the original dataset
    df = df.drop("categories", axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)
    
    # drop duplicates
    df = df[~df.duplicated(keep=False)]
    assert len(df[df.duplicated(keep=False)]) == 0
    
    return df


def save_data(df, database_filename):
    """ Save a dataframe into a database with given filename. """
    engine_path = "sqlite:///{}".format(database_filename)
    engine = create_engine(engine_path)
    df.to_sql('messages', engine, index=False)


def main():
    """ Main function that loads the csv, creates a DataFrame, cleans it and saves it into a database. """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
