import pandas as pd
import ast

class AO3DataManipulator:
    """
    A class to perform data manipulations on the AO3 Fanworks dataset.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the manipulator with a copy of the dataframe.
        """
        self.df = df.copy()

    @staticmethod
    def parse_list(val):
        """Helper to parse columns stored as string representations of Python lists."""
        if pd.isna(val):
            return []
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                # Fallback for manually parsing list-like strings
                return [item.strip() for item in val.strip('[]').replace("'", "").split(',') if item.strip()]
        return []

    def add_first_fandom_tag(self) -> 'AO3DataManipulator':
        """
        Creates a new column 'fandom_tag_1' which contains the first fandom tag
        in the list of fandom tags. If the list is empty, it assigns None.
        """
        if 'parsed_fandoms' not in self.df.columns:
            self.df['parsed_fandoms'] = self.df['Fandom Tags'].apply(self.parse_list)
            
        self.df['fandom_tag_1'] = self.df['parsed_fandoms'].apply(lambda x: x[0] if len(x) > 0 else None)
        return self

    def add_fandom_tags_count(self) -> 'AO3DataManipulator':
        """
        Creates a new column 'count_fandom_tags' which counts the number of elements
        in the fandom tags list.
        """
        if 'parsed_fandoms' not in self.df.columns:
            self.df['parsed_fandoms'] = self.df['Fandom Tags'].apply(self.parse_list)
            
        self.df['count_fandom_tags'] = self.df['parsed_fandoms'].apply(len)
        return self

    def add_archive_warning_binary(self) -> 'AO3DataManipulator':
        """
        Creates a new binary column 'archive_warning_binary' which takes value 1
        if the Warning Tags column is different from:
          - [' Creator Chose Not To Use Archive Warnings']
          - [' No Archive Warnings Apply']
        Otherwise, it takes the value 0.
        """
        # Parse warning tags if not already done
        if 'parsed_warnings' not in self.df.columns:
            self.df['parsed_warnings'] = self.df['Warning Tags'].apply(self.parse_list)

        # Check if warning list is different from the two specified lists (including leading spaces)
        self.df['archive_warning_binary'] = self.df['parsed_warnings'].apply(
            lambda x: 1 if (x != [' Creator Chose Not To Use Archive Warnings'] and 
                            x != [' No Archive Warnings Apply']) else 0
        )
        return self

    def run_all_manipulations(self) -> pd.DataFrame:
        """
        Applies all data manipulations (first fandom tag, fandom tags count, and
        archive warning binary) and returns the modified DataFrame.
        """
        return (self.add_first_fandom_tag()
                    .add_fandom_tags_count()
                    .add_archive_warning_binary()
                    .get_dataframe())

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the current state of the DataFrame."""
        return self.df
