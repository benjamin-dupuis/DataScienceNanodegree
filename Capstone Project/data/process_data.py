import numpy as np
import pandas as pd

CHANNELS = ['email', 'mobile', 'social', 'web']


def load_data():
    """Load the three json files and return corresponding DataFrames"""
    portfolio = pd.read_json('portfolio.json', orient='records', lines=True)
    profile = pd.read_json('profile.json', orient='records', lines=True)
    transcript = pd.read_json('transcript.json', orient='records', lines=True)
    return portfolio, profile, transcript


def transform_channels(channels):
    """Transform portfolio channels to bool values."""
    channels_bools = []
    for channel in CHANNELS:
        if channel in channels:
            channels_bools.append(1)
        else:
            channels_bools.append(0)
    return channels_bools


def engineer_portfolio(portfolio):
    # Change name of column id to offer_id to be coherent with the transcript DataFrame.
    portfolio['offer_id'] = portfolio['id']
    portfolio = portfolio.drop('id', axis=1)

    # Dummy the portfolio channels.
    portfolio[CHANNELS] = portfolio['channels'].apply(transform_channels).apply(pd.Series)
    portfolio = portfolio.drop('channels', axis=1)

    # Convert duration in hours.
    portfolio['duration_hours'] = portfolio['duration'].apply(lambda val: val * 24)
    portfolio = portfolio.drop('duration', axis=1)

    # Dummy the offer_type.
    portfolio = pd.get_dummies(portfolio, columns=['offer_type'])

    return portfolio


def extract_year_from_date(date):
    """Extract the year from a date in string format."""
    year = str(date)[:4]
    assert year in ['2013', '2014', '2015', '2016', '2017', '2018', '2019'], year
    return int(year)


def engineer_profile(profile):
    """Data engineering on the profile DataFrame."""
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')
    profile['year'] = profile['became_member_on'].apply(extract_year_from_date)
    profile['weekday_membership'] = profile['became_member_on'].apply(lambda date: date.weekday())

    profile = pd.get_dummies(profile, prefix=['gender', 'became_member_on'],
                             columns=['gender', 'year']).drop('became_member_on', axis=1)

    return profile


def engineer_transcript(transcript, profile):
    """Data engineering on the transcript DataFrame."""
    transcript = transcript[transcript['person'].isin(profile['id'])]
    transcript['offer_id'] = transcript['value'].apply(lambda val: val.get('offer_id',
                                                                           val.get('offer id', np.nan)))

    return transcript


def get_complete_dataframe(engineered_portfolio, engineered_profile, engineered_transcript):
    """Get complete DataFrame from engineered three DataFrames."""
    merged_transcript = engineered_transcript.merge(engineered_profile,
                                                    left_on='person',
                                                    right_on='id',
                                                    how='left').drop('person', axis=1)

    df = merged_transcript.merge(engineered_portfolio,
                                 on='offer_id',
                                 how='left').sort_values(by=['id', 'time']).reset_index(drop=True)

    return df


def is_offer_successfull(customer_df, full_df):
    """Checks if offers are successful for a given customer."""
    completed_with_success = False

    customer_successful_map = {}

    for idx, row in customer_df.iterrows():
        successful = 0
        if row.event == 'offer received':
            deadline = row.time + row.duration_hours
            next_row = full_df.loc[idx + 1]
            if next_row.event == 'offer viewed':
                next_next_row = full_df.loc[idx + 2]
                if next_next_row.time <= row.time + deadline:
                    if row.offer_type_informational == 1:
                        if next_next_row.event == 'transaction':
                            successful = 1
                    else:
                        if next_next_row.event == 'offer completed' or next_next_row.event == 'transaction':
                            successful = 1
                            completed_with_success = True

        if idx not in customer_successful_map:
            customer_successful_map[idx] = successful
            customer_successful_map[idx + 1] = successful
            customer_successful_map[idx + 2] = successful
            if completed_with_success:
                customer_successful_map[idx + 3] = successful
    return customer_successful_map


def fill_successful_offers(df):
    """Fill the offers IDs and tag them successful or not."""
    successful_map = {}
    df['successful_offer'] = np.nan
    grouped_users = df.sort_values('time').groupby('id')
    for customer, customer_df in grouped_users:
        customer_map = is_offer_successfull(customer_df, df)
        successful_map.update(customer_map)

    df['successful_offer'] = df['successful_offer'].fillna(successful_map)
    return df


def get_dataframe_for_analysis(df, portfolio):
    df_for_analysis = df.drop(['value', 'time', 'event'], axis=1) \
        .dropna(axis=0) \
        .drop_duplicates(keep='first') \
        .merge(portfolio, on='offer_id', how='inner') \
        .drop(['difficulty_x', 'reward_x'], axis=1) \
        .rename(columns={'difficulty_y': 'difficulty', 'reward_y': 'reward'})
    return df_for_analysis


def save_dataframe_to_csv(df, filepath):
    """Save a given DataFrame in a csv file."""
    df.to_csv(path_or_buf=filepath, index=False)


def engineer_final_df(df):
    """Engineer DataFrame to be ready to be used for machine learning."""
    df = df.drop(['id', 'offer_id', 'value', 'time', 'event'], axis=1).dropna(axis=0).drop_duplicates(keep='first')
    assert df.isnull().sum().sum() == 0
    assert list(df['successful_offer'].value_counts().to_dict().keys()) == [0, 1]
    return df


def main():
    """Main function to load, engineer DataFrames, and save the cleaned DataFrames in CSV format."""
    print("Loading data...")
    portfolio, profile, transcript = load_data()
    portfolio_tmp = portfolio.copy()

    print("Engineering DataFrames...")
    engineered_portfolio = engineer_portfolio(portfolio)
    engineered_profile = engineer_profile(profile)
    engineered_transcript = engineer_transcript(transcript, profile)
    complete_df = get_complete_dataframe(engineered_portfolio=engineered_portfolio,
                                         engineered_profile=engineered_profile,
                                         engineered_transcript=engineered_transcript)
    success_df = fill_successful_offers(df=complete_df)
    portfolio_tmp = portfolio_tmp.rename(columns={'id': 'offer_id'})
    df_for_analysis = get_dataframe_for_analysis(df=success_df, portfolio=portfolio_tmp)
    # Test that the DataFrame for analysis has the correct set of values for offers IDs and user IDs
    assert not (set(df_for_analysis['id']) - set(transcript['person']))
    assert not (set(df_for_analysis['offer_id']) - set(portfolio['id']))
    df_for_machine_learning = engineer_final_df(df=success_df)

    print("Saving DataFrames to CSV...")
    save_dataframe_to_csv(df_for_machine_learning, 'cleaned_data.csv')
    save_dataframe_to_csv(df_for_analysis, 'data_for_analysis.csv')
    save_dataframe_to_csv(engineered_portfolio, 'engineered_portfolio.csv')
    save_dataframe_to_csv(engineered_profile, 'engineered_profile.csv')

    print("DataFrames successfully saved!")


if __name__ == '__main__':
    main()
