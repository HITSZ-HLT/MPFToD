import json
import copy
import random

import six


def _add_adjusted_col_offsets(table):
    """Add adjusted column offsets to take into account multi-column cells."""

    if not isinstance(table, dict):
        adjusted_table = []
        for row in table:
            real_col_index = 0
            adjusted_row = []
            for cell in row:
                adjusted_cell = copy.deepcopy(cell)
                adjusted_cell["adjusted_col_start"] = real_col_index
                adjusted_cell["adjusted_col_end"] = (
                        adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"])
                real_col_index += adjusted_cell["column_span"]
                adjusted_row.append(adjusted_cell)
            adjusted_table.append(adjusted_row)
        return adjusted_table
    else:
        adjusted_table = {}
        for row_idx, row in table.items():
            real_col_index = 0
            adjusted_row = []
            for cell in row:
                adjusted_cell = copy.deepcopy(cell)
                adjusted_cell["adjusted_col_start"] = real_col_index
                adjusted_cell["adjusted_col_end"] = (
                        adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"])
                real_col_index += adjusted_cell["column_span"]
                adjusted_row.append(adjusted_cell)
            adjusted_table[row_idx] = adjusted_row
        return adjusted_table


def _get_heuristic_row_headers(adjusted_table, row_index, col_index):
    """Heuristic to find row headers."""
    row_headers = []
    row = adjusted_table[row_index]
    for i in range(0, col_index):
        if row[i]["is_header"]:
            row_headers.append(row[i])
    return row_headers


def _get_heuristic_col_headers(adjusted_table, row_index, col_index):
    """Heuristic to find column headers."""
    adjusted_cell = adjusted_table[row_index][col_index]
    adjusted_col_start = adjusted_cell["adjusted_col_start"]
    adjusted_col_end = adjusted_cell["adjusted_col_end"]
    col_headers = []
    for r in range(0, row_index):
        if not isinstance(adjusted_table, dict):
            row = adjusted_table[r]
        else:
            if r not in adjusted_table.keys():
                continue
            else:
                row = adjusted_table[r]
        for cell in row:
            if (cell["adjusted_col_start"] < adjusted_col_end and
                    cell["adjusted_col_end"] > adjusted_col_start):
                if cell["is_header"]:
                    col_headers.append(cell)

    return col_headers


def get_highlighted_subtable(table, cell_indices, with_heuristic_headers=False):
    """Extract out the highlighted part of a table."""
    highlighted_table = []
    values = []

    adjusted_table = _add_adjusted_col_offsets(table)

    for (row_index, col_index) in cell_indices:
        cell = table[row_index][col_index]
        values.append(cell["value"])
        if with_heuristic_headers:
            row_headers = _get_heuristic_row_headers(adjusted_table, row_index, col_index)
            col_headers = _get_heuristic_col_headers(adjusted_table, row_index,
                                                     col_index)
        else:
            row_headers = []
            col_headers = []

        highlighted_cell = {
            "cell": cell,
            "row_headers": row_headers,
            "col_headers": col_headers
        }
        highlighted_table.append(highlighted_cell)

    return highlighted_table, values


def get_subtable(table, cell_indices, num=8):
    """Extract out a sub part of a table."""
    subtable = []
    rows = []
    values = []

    # adjusted_table = _add_adjusted_col_offsets(table)

    for (row_index, col_index) in cell_indices:
        cell = table[row_index][col_index]
        values.append(cell["value"])
        if row_index not in rows:
            rows.append(row_index)
    # 如果表格大于8行，获取其中的8行
    if 0 not in rows:
        rows.insert(0, 0)
    if len(table) >= num:
        while len(rows) < num:
            rand_row = random.randint(1, len(table) - 1)
            if rand_row not in rows:
                rows.append(rand_row)
    else:
        for i in range(len(table)):
            if i not in rows:
                rows.append(i)

    # rows = sorted(rows)
    for r_idx in rows:
        subtable.append((r_idx, table[r_idx]))
    return subtable, values


def linearize_subtable(table, cell_indices, table_page_title,
                       table_section_title):
    """Linearize full table with localized headers and return a string."""
    table_str = ""
    if table_page_title:
        table_str += "<page_title> " + table_page_title + " </page_title> "
    if table_section_title:
        table_str += "<section_title> " + table_section_title + " </section_title> "

    table_str += "<table> "
    for (r_index, row) in table:
        row_str = "<header> " if r_index == 0 else "<row> "
        for c_index, col in enumerate(row):
            if c_index != len(row) - 1:
                item_str = col["value"] + " | "
            else:
                item_str = col["value"]
            row_str += item_str

        row_str += " </header> " if r_index == 0 else " </row> "
        table_str += row_str

    table_str += "</table>"
    # if cell_indices:
    #     assert "<highlighted_cell>" in table_str
    return table_str


def linearize_highlight_table(subtable, table_page_title, table_section_title):
    """Linearize the highlighted subtable and return a string of its contents."""
    table_str = ""
    if table_page_title:
        table_str += "<page_title> " + table_page_title + " </page_title> "
    if table_section_title:
        table_str += "<section_title> " + table_section_title + " </section_title> "
    table_str += "<table> "

    for item in subtable:
        cell = item["cell"]
        row_headers = item["row_headers"]
        col_headers = item["col_headers"]

        # The value of the cell.
        item_str = "<cell> " + cell["value"] + " "

        # All the column headers associated with this cell.
        for col_header in col_headers:
            item_str += "<col_header> " + col_header["value"] + " </col_header> "

        # All the row headers associated with this cell.
        for row_header in row_headers:
            item_str += "<row_header> " + row_header["value"] + " </row_header> "

        item_str += "</cell> "
        table_str += item_str

    table_str += "</table>"
    return table_str


def load_file(file):
    examples = []
    with open(file, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if len(examples) % 100 == 0:
                print("Num examples processed: %d" % len(examples))

            line = six.ensure_text(line, "utf-8")
            json_example = json.loads(line)
            table = json_example["table"]
            table_page_title = json_example["table_page_title"]
            table_section_title = json_example["table_section_title"]
            cell_indices = json_example["highlighted_cells"]
            # 取final sentence作为text
            text = json_example['sentence_annotations'][0]['final_sentence']

            # 获取至少num行的包含highlight cell所在行的subtable
            sub_table, entities = (get_subtable(
                table=table,
                cell_indices=cell_indices,
                num=4))

            subtable_str = (
                linearize_subtable(
                    table=sub_table,
                    cell_indices=cell_indices,
                    table_page_title=None,
                    table_section_title=None))

            examples.append({"task_id": 2, 'text': text, 'linear_table': subtable_str, 'entities': entities})
    return examples


class Parser(object):
    @staticmethod
    def load():
        train_file = 'data/totto/totto_train_data.jsonl'
        dev_file = 'data/totto/totto_dev_data.jsonl'
        print(("Reading from {},{} for dialogs".format(train_file, dev_file)))
        datas = []
        train_data = load_file(train_file)
        datas.extend(train_data)
        dev_data = load_file(dev_file)
        datas.extend(dev_data)
        return datas
