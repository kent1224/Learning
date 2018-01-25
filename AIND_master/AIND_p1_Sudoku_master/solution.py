# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:55:37 2017

@author: 14224
"""
# Strategy 1: Elimination: 依據數獨本身的規則限制刪掉數字
# Strategy 2: Only Choice: peer裡不確定的格子中，有一個可能的數子只出現在某個格子裡，則那個就一定是那個數子
# Strategy 3: Naked Twins: peer裡不確定的格子中，有兩格有相同的兩個可能性，代表這兩格就是這兩個數子，則其它peer的格子就不能有這兩個數子
# 以上就是constraint propagation
# Strategy 4: Search: 還沒解開的話，找可能性最少的格子，一個一個試可能的數字

assignments = []

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]


rows = "ABCDEFGHI"
cols = "123456789"
"""
Creating All Boxes
"""
boxes = cross(rows, cols)
"""
Creating Row Units
"""
row_units = [cross(r, cols) for r in rows]
"""
Creating Columns Units
"""
column_units = [cross(rows, c) for c in cols]
diag_units = [[],[]]
"""
Creating Diagonal Units
"""
for index, s in enumerate(rows):
    diag_units[0].append(s+cols[index])
for index, s in enumerate(reversed(rows)):
    diag_units[1].append(s + cols[index])
"""
Creating Square Units
"""
square_units = [cross(rs, cs) for rs in ["ABC", "DEF", "GHI"] for cs in ["123", "456", "789"]]
"""
Creating Enitre Unitlist
"""
unitlist = row_units + column_units + square_units + diag_units
"""
Creating Units and Peers for each box as a dictionary
"""
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    for unit in unitlist:
        naked_twin_boxes = []
        naked_twin_values = []
        for box in unit:
            box1 = box
            for other_box in unit:
                box2 = other_box
                #checking if boxes are not same, then if values are same and then if lenght is 2
                if box1 != box2 and values[box1] == values[box2] and len(values[box1]) == 2:
                    #pushing the twin boxes
                    naked_twin_boxes.append(box1)
                    naked_twin_boxes.append(box2)
                    #pushing the twin box values
                    naked_twin_values.append(values[box1])

        same_unit_boxes = []
        #find the common unit these twins belong to
        if len(naked_twin_boxes) > 1:
            for unit in unitlist:
                if naked_twin_boxes[0] in unit and naked_twin_boxes[1] in unit:
                    same_unit_boxes = unit

        if len(same_unit_boxes) > 0:
            #if there are twins and there is common unit
            for box in same_unit_boxes:
                #making sure the naked twins are not replaced
                if box not in naked_twin_boxes:
                    #replacing values of boxes if they contains any value from the naked twins
                    if naked_twin_values[0][0] in values[box]:
                        values[box] = values[box].replace(naked_twin_values[0][0], '')
                    if naked_twin_values[0][1] in values[box]:
                        values[box] = values[box].replace(naked_twin_values[0][1], '')
    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    dict = {}
    for index, box in enumerate(boxes):
        dict[box] = grid[index] if grid[index] is not "." else "123456789"
    return dict
    
    """ Result:
            'A1': '2',
            'A2': '123456789',
            'A3': '123456789',
            'A4': '123456789',
            'A5': '123456789',
            'A6': '123456789',
            'A7': '123456789',
            'A8': '123456789',
            'A9': '123456789',
            'B1': '123456789',
            'B2': '123456789',
            'B3': '123456789',
            'B4': '123456789',
            'B5': '123456789',
            'B6': '6',
            'B7': '2', 
            ... """
 
    """ In Python, it's roughly equivalent to this:
        if not condition:
            raise AssertionError()

        Try it in the Python shell:

        >>> assert True
        >>> assert False
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        AssertionError """
    """ assert 後面的敘述若是True，則繼續，否則秀出error """
    #assert len(grid) == 81, "Input grid must be a string of length 81 (9x9)"
    #return dict(zip(boxes, grid))



def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1 + max(len(values[s]) for s in boxes)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    """Eliminate values from peers of each box with a single value.
        Go through all the boxes, and whenever there is a box with a single value,
        eliminate this value from the set of values of all its peers.
        Args:
            values: Sudoku in dictionary form.
        Returns:
            Resulting Sudoku in dictionary form after eliminating values.
    """
    for box in values:
        if len(values[box]) == 1:
            for peer in peers[box]:
               values[peer] = values[peer].replace(values[box], '')
    return values


def only_choice(values):
    """Finalize all values that are the only choice for a unit.
        Go through all the units, and whenever there is a unit with a value
        that only fits in one box, assign the value to this box.
        Input: Sudoku in dictionary form.
        Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    for unit in unitlist:
        for digit in "123456789":
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                assign_value(values, dplaces[0], digit)
    return values

def reduce_puzzle(values):
    """
        Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
        If the sudoku is solved, return the sudoku.
        If after an iteration of both functions, the sudoku remains the same, return the sudoku.
        Input: A sudoku in dictionary form.
        Output: The resulting sudoku in dictionary form.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    values = reduce_puzzle(values)
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in boxes):
        return values
    box = min(s for s in boxes if len(values[s]) > 1) #找出尚未解決的box中，存在最少可能的格子
    for value in values[box]: #Search看看，直到傳回True的value # Choose one of the unfilled squares with the fewest possibilities
        new_sudoku = values.copy() #把全部複製
        new_sudoku[box] = value    #找出的那格儲存格代換成中一個
        attempt = search(new_sudoku) #解解看
        if attempt:                  #解出則傳回結果，沒解出則利用迴圈是下一個可能的值
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    solution = search(grid_values(grid))
    if solution:
        return solution
    else:
        return False


if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')