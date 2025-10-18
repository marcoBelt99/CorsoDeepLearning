if __name__ == '__main__':    
    '''
    Fare TL con una rete convoluzionale, da caricare preaddestrata.
    Devo rimuovere l'ultimo layer, aggiungere un nuovo layer, ovviamente con il numero di classi
    in base ad dataset, in questo caso è CIFAR10, quindi 10 classi, e devo addestrare solamente
    quell'ultimo layer lì.
    '''

    # Importo le librerie che mi servono
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision import models # per fare TL

    '''
    1) Definisco tutto ciò che riguarda il Dataset
    '''
    ## Trasformazioni per il preprocessing dei dati
    transform = transforms.Compose([
        transforms.Resize(224), # CIFAR10 ha immagini 32x32, ridimensiono quindi a 224x224 per ResNet
        transforms.ToTensor(), # trasformo il dato in tensore, in modo da poterlo elaborare con PyTorch
        transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) ) # Normalizzazione ==> devo andare a vedere bene 
    ])                       # quali sono le medie e le varianze di ImageNet e metterle lì. Questo perchè se voglio mantenere
                            # la conoscenza che la rete ha maturato sul mio DS grande ImageNet, per fare in modo questa conoscenza
                            # sia mantenuta e sfruttata al massimo devo fare in modo che la normalizzazione del DS, quindi come vengono
                            # normalizzati i numeri sul DS nuovo in cui vado a lavorare devono essere gli stessi che la rete ha già visto.
                            # Perchè altrimenti, magari sul DS precedente ha sempre visto dei valori di immagini che vanno da -1 a 1, e adesso
                            # se le facessi vedere immagini che vanno da 10 a 20, chiaramente la rete farebbe più fatica ad associare le vecchie
                            # caratteristiche che ha già imparato con quelle nuove, di conseguenza tutto il pre-addestramento che ho fatto è in parte
                            # inutile. Se, invece, vado a dire al mio dato "guarda, nel DS su cui la mia rete è stata pre-addestrata i dati vanno da
                            # questo valore a quest'altro, ovvero hanno questa media e questa dev. standard", allora la rete si ritrova "a casa sua",
                            # cioè si ritrova con dei dati noti su cui lavorare e di conseguenza può dare il massimo di quello che ha imparato.

    ## Caricamento del dataset CIFAR10
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # passandogli transform=transform, quello che ottengo è che ogni volta che nel mio ciclo for vado a chiamare un nuovo batch di dati, vengono
    # applicate tutte queste trasformazioni qui a tutti i dati, e io ho già direttamente il dato lavorabile con la mia rete

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2) # posso provare diverse configurazioni di batch_size
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    ## Caricamento del modello ResNet18 pre-addestrato.
    # I modelli pre-addestrati di PyTorch li trovo su torchvision. Lì trovo una lista di modelli per fare: classificazione, object detection,
    # segmentazione semantica, classificazione video, keypoint detection, etc.
    # model = models.resnet18(pretrained=True) # o gli dico pretrained=True oppure gli passo direttamente i pesi, dicendogli:
    # model = models.resnet18(weights=)

    # questo qui commentato dovrebbe essere deprecato
    # model = models.resnet18(pretrained=False) # in questo caso non la prendo pre-addestrata altrimenti ci mette un sacco di tempo
    model = models.resnet18(weights=None)

    ## Sostituisco l'ultimo layer di ResNet 18 che ha 1000 classi con quello nuovo che deve avere 10 classi di output
    # (perchè CIFAR 10 ha 10 classi di output)
    # ==> il nuovo layer che vado ad attaccare deve avere lo "stesso aggancio" di quello preesistente. Quindi, il numero di features che prende in
    # ingresso l'attuale ultimo layer di ResNet 18 e il numero di features che prende in ingresso il mio layer deve essere lo stesso.
    # Per vederlo:
    num_features = model.fc.in_features # così ho il numero esatto di features che questo layer si aspetta in ingresso
    # (Per far combiaciare il mio nuovo layer con quello vecchio, bisogna che queste in_features siano esattamente le stesse)

    # nella variabile num_features mi salvo il valore 512, dopodichè mi creo un nuovo layer Lineare FC, in cui gli dico che in ingresso
    # prende num_features e in uscita deve dare le 10 classi che desidero 
    model.fc = nn.Linear(num_features, 10) # CIFAR10 ha 10 classi
    # sto quindi dicendo che l'attuale ultimo layer di ResNet 18 (model.fc) deve essere uguale a quello nuovo.
    # TODO: Provare a vedere num_features anche a debug

    ## Definizione del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) # sposto il modello sul mio device

    '''
    Implemento una delle caratteristiche tipiche del TL: blocco tutti i layers della rete, tranne che l'ultimo.
    In PyTorch lo si fa ciclando sopra a tutti i parametri del modello (che ottengo facendo model.parameters()).
    Così facendo la mia backbone (cioè la parte che voglio tenere bloccata pre-addestrata) viene effettivamente bloccata.
    '''
    ## Congelamento dei pesi pre-addestrati (solo la fully connected layer sarà aggiornata)
    for param in model.parameters(): # per questi parametri non deve essere calcolato nessun gradiente.
        param.requires_grad = False

    # Tuttavia, così facendo mi vengono congelati anche i pesi dell'ultimo layer , ma io voglio che vengano calcolati i gradienti per tale layer
    # Quindi, solo i parametri del fully connected layer saranno aggiornati
    for param in model.fc.parameters():
        param.requires_grad = True


    ## Definisco la loss function e l'ottimizzatore
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Ovviamente gli passo i parametri del solo ultimo layer denso. Il l.r. vedo io se metterlo o non metterlo,
    # in base alle esigenze del momento



    # Funzione di addestramento del modello
    def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
        '''
        Prende in ingresso:
        - modello
        - il dataset
        - la loss
        - l'ottimizzatore
        - tipo di device
        - numero di epoche che voglio addestrarlo
        '''
        # Metto il modello in modalità di train, questo anche perchè: quando si usando reti pre-addestrate, non so layer per layer cosa c'è dentro
        # ci sono reti con tantissimi layers. 
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader: # prendo i dati di inputs e le mie labels ciclando sul data loader di train
                inputs, labels = inputs.to(device), labels.to(device) # li metto nel mio device (che sia la CPU o la GPU)

                # Azzero i gradienti (fondamentale farlo)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs) # l'immagine passa attraverso la mia rete
                loss = criterion(outputs, labels) # calcolo la loss

                # Backward pass e ottimizzazione
                loss.backward() # faccio la backpropagation calcolando i gradienti, però in questo caso solo rispetto all'ultimo layer che ho aggiunto
                optimizer.step() # ed aggiornando i pesi

                running_loss += loss.item()
            
            print(f'Epoca [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')


    # Funzione di testing
    def test_model(model, test_loader, device):
        model.eval() # modello in modalità di valutazione, perchè devo disattivare ogni forma di Dropout e cose di questo tipo
        correct = 0
        total = 0
        with torch.no_grad(): # per evitare che sia una qualsiasi forma di calcolo dei gradienti inutili in questa fase qua
            for inputs, labels in test_loader: # prendo i dati di inputs e le mie labels ciclando sul data loader di test
                inputs, labels = inputs.to(device), labels.to(device) # sposto tali dati sul mio device

                ## Faccio inferenza
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1) # calcolando il massimo vedo quale classe delle 10 ha vinto
                
                # Mi calcolo l'accuratezza
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        
        print(f'Accuratezza: {100 * correct / total:.2f}%') # moltiplicandola per 100 e mediandola (dividendo per il totale dei conti che ho fatto), 
        # così ottengo la % di accuratezza del mio modello


    ## Addestramento del modello
    train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    ## Testing del modello
    test_model(model, test_loader, device)
