import os
import requests
from tqdm import tqdm

RAW_DATA_PATH = os.path.expanduser('~')+ '/nhlapidata/raw'


class HockeyDataLoader:
    """
    Class handling all seasonal data loadings.
    """

    def __init__(self, season_years=None, base_save_path=RAW_DATA_PATH):
        if season_years is None:
            season_years = ['2016', '2017', '2018', '2019', '2020']
        assert (base_save_path.startswith(RAW_DATA_PATH))
        self.SEASONS = season_years
        self.base_save_path = base_save_path

        if not os.path.isdir(self.base_save_path):
            os.mkdir(self.base_save_path)

    def get_season_data(self, year: str) -> None:
        """
        Function using REST calls to fetch data of a whole season (regular season & playoffs). Saves resulting json in
        the path defined in self.base_save_path
        :param year: 4-digit desired season year. For example, '2017' for the 2017-2018 season.
        :return: None
        """
        # Sanity checks
        assert_year(year)

        # Get game data
        self.get_regular_season_data(year)
        self.get_playoffs_data(year)

    def get_regular_season_data(self, year: str, make_asserts: bool = True) -> None:
        """
        Function using REST calls to fetch data of a regular season of a given year. Saves resulting json in
        the path defined in self.base_save_path
        :param year: 4-digit desired season year. For example, '2017' for the 2017-2018 season.
        :param make_asserts: boolean to determine whether or not make sanity checks. False if function is called from
        get_season_data
        :return: None
        """
        if make_asserts:
            assert_year(year)

        # Regular Season game-ids
        if year == '2016':
            no_of_games = 1231  # 1230 matches in 2016, a new team was introduced after
        elif year == '2020':
            no_of_games = 869  # 868 matches in 2020 because of covid
        else:
            no_of_games = 1272

        game_numbers = ["%04d" % x for x in range(1, no_of_games)]  # 0001, 0002, .... 1271
        regular_season = [f'{year}02{game_number}' for game_number in game_numbers]

        # Get game data
        for game_id in tqdm(regular_season,total=len(regular_season), desc=f"Regular {year}-{int(year)+1} Season Matches"):
            self.get_game_data(game_id, year, make_asserts=False)

    def get_playoffs_data(self, year: str, make_asserts: bool = True) -> None:
        """
        Function using REST calls to fetch data of the playoffs of a given year. Saves resulting json in
        the path defined in self.base_save_path
        :param year: 4-digit desired season year. For example, '2017' for the 2017-2018 season.
        :param make_asserts: boolean to determine whether or not make sanity checks. False if function is called from
        get_season_data
        :return: None
        """
        if make_asserts:
            assert_year(year)

        # Playoffs game-ids.
        # eights of final
        playoffs = [f"{year}0301{matchup}{game_number}" for matchup in range(1, 9) for game_number in range(1, 8)]
        # quarter final
        playoffs.extend([f"{year}0302{matchup}{game_number}" for matchup in range(1, 5) for game_number in range(1, 8)])
        # half finals
        playoffs.extend([f"{year}0303{matchup}{game_number}" for matchup in range(1, 3) for game_number in range(1, 8)])
        # final
        playoffs.extend([f"{year}0304{1}{game_number}" for game_number in range(1, 8)])

        # Get game data
        for game_id in tqdm(playoffs, total=len(playoffs), desc=f"Playoff {year}-{int(year)+1} Season Matches"):
            self.get_game_data(game_id, year, make_asserts=False)

    def get_game_data(self, game_id: str, year: str,  make_asserts: bool = True) -> None:
        """
        Get a single game data and save it to base_save_path/game_id.json
        :param game_id: id of the game. See https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
        :param year: 4-digit desired season year. For example, '2017' for the 2017-2018 season.
        :param make_asserts: boolean to determine whether or not make sanity checks. False if function is called from
        get_season_data
        :return: None
        """
        if make_asserts:
            assert_year(year)

        # Check if file exists already
        file_path = os.path.join(self.base_save_path, f'{game_id}.json')
        if os.path.isfile(file_path):
            return

        # Request API
        response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/")

        # Write to file
        with open(file_path, 'w') as f:
            f.write(response.text)

    def acquire_data_all_seasons(self):
        """
        Fetches data for all seasons contained in self.SEASONS
        :return: None
        """
        for year in self.SEASONS:
            self.get_season_data(year)


def assert_year(year) -> None:
    """
    Simple function to assert a season year is valid.
    Extracted as a method to reduce clutter.
    :param year: Season year as a 4-letter string (ex: '2016').
    :return: None
    """
    assert (len(year) == 4)
    assert (2016 <= int(year) <= 2020)


def main():
    hockey_data_loader = HockeyDataLoader(['2016', '2017', '2018', '2019', '2020'])
    hockey_data_loader.acquire_data_all_seasons()


if __name__ == "__main__":
    main()
