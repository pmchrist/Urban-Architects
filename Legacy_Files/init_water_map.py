import numpy as np
import random
import matplotlib.pyplot as plt

def add_river(water_map, river_width, min_river_length, map_size, sea_index, land_indicator, sea_indicator, river_indicator, start_from_border):
    """Adds a river to a map with already initialised sea and lakes.
    The rivers start from some random position on the map, and flow to the sea.
    The rivers will only be added successfully if they don't encounter lakes or other rivers.
    
    Parameters:
    - water_map (array): the initialised map with sea and lakes
    - river_width (int): the width of the river to add, in cells
    - min_river_length (int): the minimal length of the river, in cells
    - map_size (int): the width and length of the map, in cells
    - sea_index (int): the column index beyond which the map is filled with sea
    - land_indicator (int/str): the label indicating whether a cell consists of land
    - sea_indicator (int/str): the label indicating whether a cell consists of sea
    - river_indicator (int/str): the label indicating whether a cell consists of river
    - start_from_border (bool): value indicating whether rivers should start from the left border of the map

    Returns:
    - return value 1 (bool): indicates whether the addition of the river was successful
    - return value 2 (None or array): if the addition was successful, this value contains the map with the river added. Otherwise, this value is None.

    """

    # the river width should be an odd value, for successful computation
    assert river_width % 2 == 1

    # create a mask array for the cells of the river
    # if a cell belongs to the river, we'll change the value of that cell in the mask to True
    # if the addition of the river was successful, we use this mask to add the river to the map
    river_mask = np.full(water_map.shape, False, dtype=bool)

    # choose a random location for the start of the river
    river_start_row = random.randint(0, map_size)
    if start_from_border:
        river_start_column = 0
    else:
        # if the river starts at a random column, ensure that this is at least the minimal length of the river away from the border
        river_start_column = random.randint(0, sea_index - min_river_length)
        
    # get the current cell values at the proposed start of the river
    river_start = water_map[river_start_row - round((river_width - 1) / 2):river_start_row + round((river_width - 1) / 2) + 1, river_start_column]

    # if the start contains values indicating something else than land, we don't accept it
    # return that the addition was unsuccessful
    if not (river_start == land_indicator).all():
        return False, None

    # if we reach this point, the last check was passed
    # accept the proposed river start by setting the cell values in the river mask to true
    river_mask[river_start_row - round((river_width - 1) / 2):river_start_row + round((river_width - 1) / 2) + 1, river_start_column] = True

    # rename the variable name for the river column index for clarity
    next_river_column = river_start_column

    # keep adding river elements until the sea is reached (success) or until the river encounters a lake or other river (failure)
    while True:
        # generate a new river element that neighbours at least one cell of the prvious element
        next_river_row = random.randint(river_start_row - round((river_width - 1) / 2), river_start_row + round((river_width - 1) / 2))
        next_river_column += 1

        current_river_location = water_map[next_river_row - round((river_width - 1) / 2):next_river_row + round((river_width - 1) / 2) + 1, next_river_column]

        # if the sea is reached, the river addition was successful
        if sea_indicator in current_river_location:
            break

        # if a cell containing something else than land (or sea, but that case we already checked) is encountered, we don't accept it
        # return that the addition was unsuccessful
        if not (current_river_location == land_indicator).all():
            return False, None
        
        # if no sea, lakes or other rivers were encountered, add the river element to the river mask
        river_mask[next_river_row - round((river_width - 1) / 2):next_river_row + round((river_width - 1) / 2) + 1, next_river_column] = True

    # add the river to the water map
    water_map[river_mask] = river_indicator

    return True, water_map


def init_water_map(size, sea_fraction, num_lakes, lake_size, num_rivers):
    """Initializes a map with a sea and possible lakes and rivers.

    Parameters:
    - size (int): indicates the size N of the NxN map
    - sea_fraction (float): indicates the fraction of the map that should be filled with sea
    - num_lakes (int): indicates the number of lakes to be added to the map
    - lake_size (int): indicates the radius in cells of the lakes
    - num_rivers (int): indicates the number of rivers to be added to the map
    
    Returns:
    - water_map: a 2D numpy array filled with labels indicating the type of each cell: land, fresh water (rivers and lakes) or salt water (sea)
    """

    # choose a random seed for repeatability of simulations
    random.seed(2)

    # initialise the map with only zeros (indicating land)
    land_indicator = 0
    if land_indicator == 0:
        water_map = np.zeros((size, size))
    else:
        water_map = np.full((size, size), land_indicator, type(land_indicator))

    # determine beyond which column index the map should be filled with sea on the right
    sea_index = round((1 - sea_fraction) * size)

    # set the label indicating the sea
    sea_indicator = -2

    # fill the desired fraction of the map with sea
    column_indices, row_indices = np.meshgrid(np.arange(size), np.arange(size))
    mask = column_indices >= sea_index
    water_map[mask] = sea_indicator

    # set the label indicating a lake
    lake_indicator = -1

    # add the desired number of lakes
    for lake in range(num_lakes):

        # generate a random lake position that is at least the radius of the lake away from the sea to prevent overlap
        lake_center_column = random.randint(lake_size, sea_index - lake_size)
        lake_center_row = random.randint(lake_size, size)

        # add a lake with the desired radius at the generated position
        row_indices, column_indices = np.indices(water_map.shape)
        mask = (row_indices - lake_center_row) ** 2 + (column_indices - lake_center_column) ** 2 <= lake_size ** 2
        water_map[mask] = lake_indicator
    
    # set the label indicating a river
    river_indicator = -1

    # add the desired number of rivers
    for river in range(num_rivers):

        river_addition_succesful, water_map_with_river = add_river(water_map, 3, 10, size, sea_index, land_indicator, sea_indicator, river_indicator, True)

        # adding the river might fail if the river encounters a lake or other river
        # keep track here of the number of retries
        river_tries = 0

        # keep trying to add a river if this fails, up to 10 times
        while (river_addition_succesful == False) and (river_tries < 10):
            river_tries += 1
            river_addition_succesful, water_map_with_river = add_river(water_map, 3, 10, size, sea_index, land_indicator, sea_indicator, river_indicator, True)

        # if the river takes more than 10 tries, assume that it can't be added and throw an exception
        if river_addition_succesful == False:
            raise Exception()
    
        # if the river was successfully added, set the water map to the map with the added river
        water_map = water_map_with_river

    return water_map

if __name__=="__main__":
    water_map = init_water_map(100, 0.4 , 3, 5, 2)
    
    figure = plt.imshow(water_map, cmap="Blues")
    plt.colorbar(figure)
    plt.show()
