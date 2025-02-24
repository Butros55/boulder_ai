import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class ImageProcessor {
  static final String _serverUrl = 'http://127.0.0.1:5000/process';
  static final storage = FlutterSecureStorage();

  static Future<Map<String, dynamic>> processImage(dynamic image) async {
    http.MultipartRequest request = http.MultipartRequest(
      'POST',
      Uri.parse(_serverUrl),
    );

    final token = await storage.read(key: 'jwt_token');
    if (token != null) {
      request.headers["Authorization"] = "Bearer $token";
    }

    if (kIsWeb) {
      if (image is! Uint8List) {
        throw Exception('Auf Web muss image vom Typ Uint8List sein.');
      }
      request.files.add(
        http.MultipartFile.fromBytes('image', image, filename: 'image.jpg'),
      );
    } else {
      if (image is! File) {
        throw Exception(
          'Auf mobilen Plattformen muss image vom Typ File sein.',
        );
      }
      request.files.add(await http.MultipartFile.fromPath('image', image.path));
    }

    var response = await request.send();
    if (response.statusCode == 200) {
      final bytes = await response.stream.toBytes();
      final responseString = utf8.decode(bytes);
      final Map<String, dynamic> result = json.decode(responseString);
      return result;
    } else {
      throw Exception(
        'Bildverarbeitung fehlgeschlagen: ${response.statusCode}',
      );
    }
  }
}
