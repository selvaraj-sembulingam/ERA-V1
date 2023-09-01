# Transformers from scratch

## Transformer

<img src="https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/362f1888-8953-4abb-a2a8-1d9bcdebae79" width=20% height=20%>



## Folder Structure
```
└── logs/
    └── events.out.tfevents.1693586879.a946da573979.29.0
    └── events.out.tfevents.1693597748.a946da573979.29.1
└── README.md
└── config.py
└── datamodule.py
└── dataset.py
└── litmodel.py
└── main.ipynb
└── model.py
└── tokenizer_en.json
└── tokenizer_it.json
└── train.py
```

## Train Loss
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/d5210fab-9608-4b52-84ac-95ee32094dd8)


## Train logs with Loss
```
Epoch: 1
Train Loss: 6.236441135
----------
    SOURCE: 'Papa!' exclaimed Kitty and closed his mouth with her hands.
    TARGET: — Papà — gridò Kitty e gli chiuse la bocca con le mani.
 PREDICTED: — Ma , — disse Levin , e si disse , e si fermò .
----------
    SOURCE: "Thank you: I shall do: I have no broken bones,--only a sprain;" and again he stood up and tried his foot, but the result extorted an involuntary "Ugh!"
    TARGET: — Grazie, non ho nulla di rotto, si tratta di una storta. Volle provarsi un'altra volta a camminare, ma involontariamente gettò un grido.
 PREDICTED: — Sì , — disse , — mi disse , — mi disse , — e la signora , — e la signora , e la signora , e la signora .
Validation: 0it [00:00, ?it/s]

Epoch: 2
Train Loss: 5.543663979
----------
    SOURCE: At the very moment that Vronsky thought it time to pass Makhotin, Frou-Frou, understanding what was in his mind, without any urging, considerably increased her speed and began to draw nearer to Makhotin on the side where it was most advantageous to pass him – the side of the rope, Makhotin would not let her pass that side.
    TARGET: Proprio nel momento in cui Vronskij pensava di oltrepassare Machotin, Frou-Frou stessa, intuendone il pensiero, senza essere stimolata, accelerò notevolmente il galoppo, e cominciò ad avvicinarsi a Machotin dal lato più conveniente, cioè rasente la corda. Machotin però non lasciava andare la corda.
 PREDICTED: In quel momento , Levin , che era stato di lui , Levin , che , Levin , che , Levin , che , in quel momento , che , in quel momento , in un ’ altra parte , che , in cui , in cui , in un ’ altra volta , che non si era stato in un ’ altra parte di cui si era stato in un ’ altra parte di cui si era stato di un ’ altra parte di cui non si era stato in un ’ altra parte di cui si era stato di cui , e di cui il suo modo di cui si era stato in un ’ altra parte di cui si era stato di un ’ altra parte di cui il suo modo di un ’ altra parte di cui , e , e , e , e , e che , e , e , e di un ’ altra parte di cui il suo momento , e di cui il suo momento , e di cui il suo momento , e che il suo modo di un momento , e di cui si era stato di cui si era stato di cui il suo modo di cui si era stato di cui si era stato di cui era stato di cui , e di cui si era stato di cui , e di cui , e , e , e di cui , e di cui si era stato di cui , e di cui si era stato di cui si era stato di cui si era stato di cui , e , e che , e che , e che , e che , e che , e che , e che , e che , e che , e che , e che , e , e , e che , e , e , e che , e che , e , e , e che , e che , e , e , e , e , e , e , e
----------
    SOURCE: It was old and beginning to decay.
    TARGET: Ma ormai anch’essa era cadente e ammuffita.
 PREDICTED: Era un ’ altra cosa .
Validation: 0it [00:00, ?it/s]

Epoch: 3
Train Loss: 5.191849709
----------
    SOURCE: 'You will allow me to listen?' he asked.
    TARGET: — Mi permettete di ascoltare? — chiese.
 PREDICTED: — Avete bisogno di me ? — domandò .
----------
    SOURCE: "Well?" I said, as he again paused--"proceed."
    TARGET: — Bene, — dissi, quando tacque, — continuate.
 PREDICTED: — Ebbene , — disse , — e io sono felice .
Validation: 0it [00:00, ?it/s]

Epoch: 4
Train Loss: 4.898487568
----------
    SOURCE: I don't very well know what I did with my hands, but he called me "Rat!
    TARGET: Non so dire quello che io facessi con le mani, ma John mi chiamava: "Talpa!
 PREDICTED: Non so che cosa mi , ma mi misi a piangere , ma mi disse :
----------
    SOURCE: So I asked him if he would, and if we might venture over in her. “Yes,” he said, “we venture over in her very well, though great blow wind.”
    TARGET: Gli chiesi pertanto se voleva e se dovevamo arrischiarci sovr’essa. — «Sì, rispose, potere e volere, anche se soffiar vento grande .»
 PREDICTED: E se fosse andato a me , e se fosse andato a casa , se ne , , , , il fuoco , disse : — « « « , come se ne , come se ne , come un vento di vento .
Validation: 0it [00:00, ?it/s]

Epoch: 5
Train Loss: 4.631300926
----------
    SOURCE: He was dressed now: he still looked pale, but he was no longer gory and sullied.
    TARGET: Era vestito e mi parve debolissimo, ma su di lui non c'era nessuna traccia di sangue.
 PREDICTED: Egli era troppo forte , ma non era più forte , ma non era più bello .
----------
    SOURCE: Having lived most of his life in the country and in close contact with the peasants, Levin always felt, at this busy time, that this general stimulation of the peasants communicated itself to him.
    TARGET: Avendo vissuto la maggior parte della sua vita in campagna e in rapporti intimi con la gente di campagna, Levin, nel periodo di lavoro, sentiva sempre che quella generale eccitazione del popolo si comunicava anche a lui.
 PREDICTED: sempre più la vita di vita e di campagna , Levin si sentiva sempre più forte , che Levin sentiva che , in questo tempo , in questo tempo , era con l ’ azienda dell ’ azienda .
Validation: 0it [00:00, ?it/s]

Epoch: 6
Train Loss: 4.372273445
----------
    SOURCE: "You see now, my queenly Blanche," began Lady Ingram, "she encroaches. Be advised, my angel girl--and--"
    TARGET: — Vedete, mia regale Bianca, essa diventa sempre più esigente.
 PREDICTED: — Avete fatto bene , signorina , — continuò Bianca , — venite a mio zio , e mi ha detto :
----------
    SOURCE: "Well, I would rather die yonder than in a street or on a frequented road," I reflected. "And far better that crows and ravens--if any ravens there be in these regions--should pick my flesh from my bones, than that they should be prisoned in a workhouse coffin and moulder in a pauper's grave."
    TARGET: — Ebbene, — dissi a me stessa, — preferisco morir qui piuttosto che su una strada frequentata, e se vi sono corvi nel vicinato preferisco che essi si pascano delle mie carni piuttosto che sapere il mio corpo rinchiuso nella bara di un ospedale e sepolto nella fossa dei poveri.
 PREDICTED: — Ebbene , vorrei esser buona , o in mezzo a un paese o in mezzo a una casa , — mi disse , — e se mi sarei stato meglio che mi in Inghilterra , e se mi sarei stato un di in Inghilterra , e se mi in una .
Validation: 0it [00:00, ?it/s]

Epoch: 7
Train Loss: 4.123306274
----------
    SOURCE: Levin turned round.
    TARGET: Levin si voltò a guardare.
 PREDICTED: Levin si voltò .
----------
    SOURCE: Now hear how the deacon will roar, "Wives, obey your husbands"!'
    TARGET: Su, senti come urla il diacono: “che tema suo marito”.
 PREDICTED: Ora , come un sorriso , i vostri diritti , le quali la vostra volontà .
Validation: 0it [00:00, ?it/s]

Epoch: 8
Train Loss: 3.876793146
----------
    SOURCE: "Mr. Wood is in the vestry, sir, putting on his surplice."
    TARGET: — Il signor Wood è giunto e si veste.
 PREDICTED: — Il signor Rochester è in mezzo al sicuro , signore , e nel suo sangue .
----------
    SOURCE: 'WELL, HAVE YOU HAD A GOOD TIME?' she asked, coming out to meet him with a meek and repentant look on her face.
    TARGET: — Be’, c’è stata allegria? — chiese lei, uscendogli incontro con un’espressione colpevole e mansueta nel viso.
 PREDICTED: — Be ’, tu , sei stato fatto da te ? — ella disse , avvicinandosi a lui con un gesto energico , con un gesto spaventato e preoccupato .
Validation: 0it [00:00, ?it/s]

Epoch: 9
Train Loss: 3.643882513
----------
    SOURCE: 'Yes, yes!'
    TARGET: — Sì, sì.
 PREDICTED: — Sì , sì !
----------
    SOURCE: What are you?
    TARGET: Cosa mai siete?
 PREDICTED: Che cosa avete ?
Validation: 0it [00:00, ?it/s]

Epoch: 10
Train Loss: 3.421863794
----------
    SOURCE: [Long argument between Harris and Harris's friend as to what Harris is really singing.
    TARGET: (Lunga discussione fra Harris e l’amico di Harris su ciò che Harris realmente canti.
 PREDICTED: di Harris e di , come Harris di Harris , che è veramente veramente .
----------
    SOURCE: What an appetite I get in the country, wonderful!
    TARGET: Che appetito m’è venuto in campagna, un prodigio!
 PREDICTED: Che gioia ho fatto in campagna , in campagna !
```
