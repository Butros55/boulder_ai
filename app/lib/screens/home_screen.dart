import 'dart:convert';
import 'dart:io';
import 'package:boulder_ai/util/image_processor.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  HomeScreenState createState() => HomeScreenState();
}

class HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  int _totalAnalyses = 0;

  @override
  void initState() {
    super.initState();
    _fetchTotalAnalyses();
  }

  final storage = FlutterSecureStorage();

  Future<String?> getToken() async {
    return await storage.read(key: 'jwt_token');
  }

  Future<void> _fetchTotalAnalyses() async {
    try {
      final token = await getToken();
      if (token == null) {
        debugPrint(
          "Kein Token gefunden, Analysen können nicht geladen werden.",
        );
        return;
      }
      final response = await http.get(
        Uri.parse("http://127.0.0.1:5000/analyses"),
        headers: {"Authorization": "Bearer $token"},
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _totalAnalyses = data['total'] as int;
        });
      } else {
        debugPrint("Fehler beim Laden der Analysen: ${response.statusCode}");
      }
    } catch (e) {
      debugPrint("Fehler beim Abrufen der Analysen: $e");
    }
  }

  Future<void> _takePicture() async {
    try {
      final XFile? picture = await _picker.pickImage(
        source: ImageSource.camera,
      );
      if (picture != null) {
        dynamic imageData;
        if (kIsWeb) {
          imageData = await picture.readAsBytes();
        } else {
          imageData = File(picture.path);
        }
        final result = await ImageProcessor.processImage(imageData);
        Navigator.pushNamed(
          context,
          '/result',
          arguments: result,
        ).then((_) => _fetchTotalAnalyses());
      }
    } catch (e) {
      debugPrint('Fehler beim Aufnehmen des Fotos: $e');
    }
  }

  Future<void> _pickImage() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
      );
      if (pickedFile != null) {
        dynamic imageData;
        if (kIsWeb) {
          imageData = await pickedFile.readAsBytes();
        } else {
          imageData = File(pickedFile.path);
        }
        final result = await ImageProcessor.processImage(imageData);
        Navigator.pushNamed(
          context,
          '/result',
          arguments: result,
        ).then((_) => _fetchTotalAnalyses());
      }
    } catch (e) {
      debugPrint('Fehler beim Auswählen eines Bildes: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    // Farben / Thema anpassen
    final Color backgroundColor = const Color(0xFF1C1C1E);
    final Color cardColor = const Color(0xFF2C2C2E);
    final Color accentColor = const Color(0xFF7F5AF0);
    final Color textColor = Colors.white;

    return Scaffold(
      backgroundColor: backgroundColor,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Titel + evtl. Icon / Avatar
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Boulder AI',
                    style: TextStyle(
                      color: textColor,
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  CircleAvatar(
                    backgroundColor: cardColor,
                    child: Icon(Icons.person, color: textColor),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              // Untertitel oder kurze Beschreibung
              Text(
                'Willkommen! \nErkenne Boulder-Griffe per Foto oder Bild-Upload.',
                style: TextStyle(
                  color: textColor.withOpacity(0.8),
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 40),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: cardColor,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  children: [
                    Text(
                      'Dein Fortschritt',
                      style: TextStyle(
                        color: textColor,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Bisher $_totalAnalyses Boulder analysiert',
                      style: TextStyle(
                        color: textColor.withOpacity(0.7),
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 40),
              // Buttons zum Aufnehmen / Hochladen
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _buildOptionButton(
                    label: 'Foto aufnehmen',
                    icon: Icons.camera_alt_rounded,
                    onTap: _takePicture,
                    accentColor: accentColor,
                    textColor: textColor,
                  ),
                  _buildOptionButton(
                    label: 'Bild hochladen',
                    icon: Icons.photo_library_rounded,
                    onTap: _pickImage,
                    accentColor: accentColor,
                    textColor: textColor,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildOptionButton({
    required String label,
    required IconData icon,
    required VoidCallback onTap,
    required Color accentColor,
    required Color textColor,
  }) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 8),
          padding: const EdgeInsets.symmetric(vertical: 16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [accentColor.withOpacity(0.8), accentColor],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            children: [
              Icon(icon, color: textColor),
              const SizedBox(height: 8),
              Text(
                label,
                style: TextStyle(color: textColor, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
