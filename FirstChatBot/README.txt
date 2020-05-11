This (final) project was made on the coursera Natural Language Processing course. The code is a little bit obsolete because I used old version of chatterbot, tensorflow, numpy but it is effective (you can test it see below).

This is the code of the Chatbot that I implement on Telegram. He can discuss (weirdly) a little bit thanks to the chatterbot library.
But its main goal is to help you with your programming issues. 
You just have have to explain in few words your problem about language programming (like R, python etc) and Bougzerbot will lead you to a StackOverflow post that already solve it.

I used a Sarspace embedding trained on Stackoverflow data for the topic modeling task. The Stackoverflow data is store by topic. And then I used the vector representation of the question to match it (cosine similarity) with an existing post on StackOverflow.

To implement it I used docker and tmux to maintain the bot active and AWS to host the program. 

You can reach the bot on Telegram @BougzerBot.


