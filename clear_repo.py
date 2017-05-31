import shutil

WARNING = """
Warning: Delete all generated folders!
"""
print(WARNING)
shutil.rmtree('tmp', ignore_errors=True)
shutil.rmtree('log', ignore_errors=True)
shutil.rmtree('tf_log', ignore_errors=True)
