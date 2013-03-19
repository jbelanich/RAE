from util import *
from RAE import *

params = genRandParams()
c1 = RAETreeNode(params,genRandWord())
c2 = RAETreeNode(params,genRandWord())

c3 = RAETreeNode(params,c1,c2)

print c3.p
print "\n"
print c3.reconError
print "\n"
print shape(c3.p)