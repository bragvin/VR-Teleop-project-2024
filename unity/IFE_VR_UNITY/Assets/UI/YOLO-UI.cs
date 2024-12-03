using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class YoloReceiver : MonoBehaviour
{
    TcpClient client;
    NetworkStream stream;

    void Start()
    {
        client = new TcpClient("127.0.0.1", 8080);
        stream = client.GetStream();
    }

    void Update()
    {
        if (stream.DataAvailable)
        {
            byte[] buffer = new byte[client.ReceiveBufferSize];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            string json = Encoding.UTF8.GetString(buffer, 0, bytesRead);

            // Parse JSON og vis objektene som overlay
            Debug.Log(json);
        }
    }

    private void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}