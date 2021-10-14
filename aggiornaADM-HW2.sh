#!/bin/bash

#Questo script aggiorna la directory di github (~/Documenti/GitHub_projects/ADM-HW2)

#Si sposta nella directory di git
cd ~/Documenti/GitHub_projects/ADM-HW2

#Controlla che non ci siano aggiornamenti e poi carica le modifiche fatte nello stage		
git pull origin main ; git add * 

#Controlla che non siano stati eliminati dei file dalla directory		 
log=`git commit -m "aggiornato" | awk '/deleted:/{print $2}'`

#se li trova li elimina da git		
for i in $log
	do
		git rm $i; git commit -m "aggiornato"
	
done

#aggiorna la directory 		
<<<<<<< HEAD
<<<<<<< HEAD
git push origin ame
=======
git push origin #branch name
>>>>>>> 921003dc3d7ea2cd96da0623033ecae0c1d39e10
=======

git push origin ame

>>>>>>> 3b5501ccc8d1303fd1fd9bee894c14932507d099

