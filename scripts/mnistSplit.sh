tar -xzvf ../dataset/mnist.tar.gz -C ../dataset/
java -jar ../Split.jar ../dataset mnist 60000 2
