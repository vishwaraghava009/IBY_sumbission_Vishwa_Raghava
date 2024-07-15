import http.server
import socketserver

PORT = 8000
DIRECTORY = "/home/iby_vishwa/Documents/SillyTavern-extras/modules/text_to_speech/alltalk_tts/outputs"

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

if __name__ == "__main__":
    handler = CORSRequestHandler
    handler.extensions_map.update({
        '.json': 'application/json',
    })
    httpd = socketserver.TCPServer(("", PORT), handler)

    print(f"Serving HTTP on 0.0.0.0 port {PORT} (http://0.0.0.0:{PORT}/) ...")
    httpd.serve_forever()
