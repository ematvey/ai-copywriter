def debug_hook():
  import sys, ipdb, traceback
  def info(type, value, tb):
      traceback.print_exception(type, value, tb)
      print()
      ipdb.pm()
  sys.excepthook = info
  print('ipdb hook injected')