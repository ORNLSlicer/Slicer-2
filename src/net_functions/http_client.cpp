////! \author Jingyang Guo

//#include "net_functions/http_client.h"

//using namespace ORNL;

//HTTPClient::HTTPClient()
//{
//    manager = new QNetworkAccessManager(this);
//    connect(manager, SIGNAL(finished(QNetworkReply*)), this, SLOT(replyFinished(QNetworkReply*)));
//}

//HTTPClient::HTTPClient(QString url)
//{
//    this->setDefaultUrl(url);
//    manager = new QNetworkAccessManager(this);
//    connect(manager, SIGNAL(finished(QNetworkReply*)), this, SLOT(replyFinished(QNetworkReply*)));
//}

//void HTTPClient::setDefaultUrl(QString url)
//{
//    default_url = url;
//}

//void HTTPClient::get()
//{
//    this->get(default_url);
//}

//void HTTPClient::get(QString url)
//{
//    reply = manager->get(QNetworkRequest(QUrl(url)));
//    connect(reply, SIGNAL(readyRead()), this, SLOT(slotReadyRead()));
//}

//void HTTPClient::slotReadyRead()
//{
//    QByteArray bytes = reply->readAll();
//    qDebug() << bytes;
//}

//void HTTPClient::post(QString postValue)
//{
//    this->post(default_url, postValue);
//}

//void HTTPClient::post(QString url, QString postValue)
//{
//    reply = manager->post(QNetworkRequest(QUrl(url)), postValue.toUtf8());
//    connect(reply, SIGNAL(readyRead()), this, SLOT(slotReadyRead()));
//}

//void HTTPClient::replyFinished(QNetworkReply *resp){
//    qDebug() << "Reply finished.";
//}
