import nbclean as nbc
path_original_notebook = r".\task2.ipynb"
path_save = r'.\task2_cleaned.ipynb'
ntbk = nbc.NotebookCleaner(path_original_notebook)
ntbk.clear('output')
# Now we'll save the notebook to inspect
ntbk.save(path_save + 'test_notebook_cleaned.ipynb')