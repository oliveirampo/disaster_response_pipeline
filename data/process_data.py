"""ETL pipeline preparation.

Input files read:
    - data/messages.csv
    - data/categories.csv

Methods:
    read_aata()
"""


from sqlalchemy import create_engine
import pandas as pd
import sys
import os


def read_data(message_file, categories_file):
    """Reads two csv files and converts each one to a pandas DataFrame.
    Files:
        - messages.csv
        - categories.csv

    :return:
        df: (pandas DataFrame) Merged table.
    """

    if not os.path.exists(message_file):
        print('\nNo such file: {}'.format(message_file))
        sys.exit(1)

    if not os.path.exists(categories_file):
        print('\nNo such file: {}'.format(categories_file))
        sys.exit(1)

    messages = pd.read_csv(message_file)
    categories = pd.read_csv(categories_file)

    # Merge datasets.
    df = pd.merge(messages, categories, on='id', how='left')

    return df


def clean_data(df):
    """Returns dataFrame with cleaned data.

    :param df: (pandas DataFrame)
    :return:
        df: (pandas DataFrame) Cleaned data.
    """

    # Split categories into separate category columns
    categories = split_categories(df)

    # Convert category values to just numbers 0 or 1.
    print('Cleaning data...')
    categories = convert_to_binary(categories)

    # remove columns with only one value
    one_value_columns = [
        column for column in categories.columns if len(categories[column].unique()) == 1
    ]
    categories = categories.drop(one_value_columns, axis=1)

    # Replace categories column in df with new category columns.
    df = update_category_values(df, categories)

    # Remove duplicates.
    df = remove_duplicates(df)

    return df


def split_categories(df):
    """Splits categories into separate category columns.
    Each row in 'categories' column has 36 categories.

    :param df: (pandas DataFrame).
    :return:
        categories: (pandas DataFrame) Table with categories split.
    """

    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    return categories


def convert_to_binary(df):
    """Convert category values to just numbers 0 or 1.

    :param df: (pandas DataFrame) Table with categories.
    :return:
        df: (pandas DataFrame) Table with categories values: 0 or 1.
    """

    for column in df:
        # set each value to be the last character of the string
        # convert column from string to numeric
        df[column] = df[column].astype(str).str[-1].astype(int)

    return df


def update_category_values(df, categories):
    """Replace categories column in df with new category columns.

    :param df: (pandas DataFrame) Table with messages and categories not formatted.
    :param categories: (pandas DataFrame) Table with categories.
    :return:
        df: (pandas DataFrame) Table with messages and formatted categories.
    """

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    return df


def remove_duplicates(df):
    """Removes duplicated values from pandas DataFrame.

    :param df: (pandas DataFrame)
    :return:
        df: (pandas DataFrame) Table without duplicated rows.
    """

    # check number of duplicates
    isAnyElementTrue = df.duplicated().any()
    if isAnyElementTrue:
        print('\tThere are duplicated values.')

    # drop duplicates
    df = df.copy().drop_duplicates()

    # related column has values 0, 1 or 2, replace 2 -> 1
    df['related'].replace(2, 1, inplace=True)

    # check number of duplicates
    isAnyElementTrue = df.duplicated().any()
    if isAnyElementTrue:
        print('\tThere are duplicated values.')
    else:
        print('\tDuplicated values have been removed.')

    return df


def save_dataframe_to_sql_db(df, database_filepath):
    """Save the clean dataset into an sqlite database.

    :param df: (pandas DataFrame) Table with messages and categories.
    :param database_filepath: (str) Path where database will be saved.
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        # Load datasets.
        df = read_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        # Save the clean dataset into an sqlite database.
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_dataframe_to_sql_db(df, database_filepath)
        print(df.head(5))

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