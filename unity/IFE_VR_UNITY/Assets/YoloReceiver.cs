using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class YoloReceiver1 : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;

    void Start()
    {
        // Koble til Python-serveren
        client = new TcpClient("127.0.0.1", 8080);
        stream = client.GetStream();
        Debug.Log("Koblet til Python-server");
    }

    void Update()
    {
        if (stream != null && stream.DataAvailable)
        {
            // Les data fra serveren
            byte[] buffer = new byte[1024];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            string json = Encoding.UTF8.GetString(buffer, 0, bytesRead);

            Debug.Log($"Mottatt data: {json}");

            // Eventuelt: Pass JSON-data til en overlay-controller
            // FindObjectOfType<OverlayController>()?.DrawBoxes(json);
        }
    }

    void OnApplicationQuit()
    {
        // Lukk forbindelsen når applikasjonen avsluttes
        stream?.Close();
        client?.Close();
    }
}