s=open('main.tex',encoding='utf8').read()
def rep(x,y):
    global s
    assert x in s, x[:50]; s=s.replace(x,y)
rep('kulpa1977,coeurjolly2004,lindblad2005','kulpa1977,lindblad2005')
rep(' (symmetric elsewhere)','')
rep('The Medical Segmentation Decathlon lung task','The Medical Segmentation Decathlon lung set')
rep(' Its annotations are drawn slice by slice, so re-digitizing them changes the object itself. To isolate the estimator,',' As its slice-wise annotations change when re-digitized, to isolate the estimator')
open('main.tex','w',encoding='utf8').write(s)
