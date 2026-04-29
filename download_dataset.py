"""
Dataset Downloader
==================
Downloads the SMS Spam Collection dataset from UCI ML Repository.
"""

import urllib.request
import zipfile
import os
import shutil


def download_sms_spam_dataset():
    """
    Download and extract the SMS Spam Collection dataset.
    """
    print("Downloading SMS Spam Collection dataset...")
    
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam.zip"
    
    try:
        # Download the zip file
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
        # Extract the zip
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("sms_spam_temp")
        
        # Find and rename the data file
        for root, dirs, files in os.walk("sms_spam_temp"):
            for file in files:
                if file == "SMSSpamCollection":
                    src = os.path.join(root, file)
                    shutil.copy(src, "spam.csv")
                    print(f"Dataset saved as 'spam.csv'")
                    break
        
        # Cleanup
        os.remove(zip_path)
        shutil.rmtree("sms_spam_temp")
        
        print("Dataset ready!")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        print("2. Download the dataset zip file")
        print("3. Extract 'SMSSpamCollection' and rename it to 'spam.csv'")
        print("4. Place it in the same folder as spam_detector.py")
        return False


def create_sample_dataset():
    """
    Create a small sample dataset for testing when the real dataset is unavailable.
    """
    sample_data = """ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham\tU dun say so early hor... U c already then say...
ham\tNah I don't think he goes to usf, he lives around here though
spam\tFreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
ham\tEven my brother is not like to speak with me. They treat me like aids patent.
ham\tAs per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
spam\tWINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
spam\tHad your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030
ham\tI'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.
spam\tSIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info
spam\tURGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18
ham\tI've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.
ham\tI HAVE A DATE ON SUNDAY WITH WILL!!
spam\tXXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL
ham\tOh k...i'm watching here:)
ham\tEh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.
spam\tWANNA HAVE FUN? CALL 08000839402 FOR LIVE CHAT WITH GIRLS. JUST 10P PER MIN. CALL NOW!
ham\tFine if that's the way u feel. That's the way its gota b
spam\tCongratulations! You've been selected for a FREE iPhone 14. Click here to claim: http://free-iphone.scam.com
ham\tAre you free this weekend? Want to grab coffee?
spam\tYou have been awarded a cash prize of $5000. To claim, send your bank details to claim@prize.com
ham\tDon't forget to bring the documents tomorrow morning.
spam\tACT NOW! Limited time offer. Buy 2 get 3 FREE. Call 1-800-SCAM-NOW to order!
ham\tSee you at the gym at 6pm?
spam\tYour loan has been approved! Get $10,000 instantly. No credit check. Call now!
ham\tCan you send me the report by end of day?
ham\tHappy birthday! Hope you have a wonderful day!
spam\tCLAIM YOUR FREE GIFT NOW! You are our lucky winner. Text GIFT to 5555."""
    
    with open("spam.csv", "w", encoding="utf-8") as f:
        f.write(sample_data)
    
    print("Sample dataset created as 'spam.csv' (30 messages for testing)")
    print("For full results, download the real dataset with: python download_dataset.py")


if __name__ == "__main__":
    success = download_sms_spam_dataset()
    if not success:
        print("\nCreating sample dataset instead...")
        create_sample_dataset()
