python -m allennlp.run train

Configure: 

python -m allennlp.run configure --include-package jigsaw

Train:

python -m allennlp.run train -s ./result --include-package jigsaw ./jigsaw/config.jsonp