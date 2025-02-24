import 'dart:io';
import 'package:boulder_ai/util/image_processor.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  HomeScreenState createState() => HomeScreenState();
}

class HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();

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
        if (!mounted) return;
        Navigator.pushNamed(context, '/result', arguments: result);
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
        if (!mounted) return;
        Navigator.pushNamed(context, '/result', arguments: result);
      }
    } catch (e) {
      debugPrint('Fehler beim Auswählen eines Bildes: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    // Farben / Thema anpassen
    final Color backgroundColor = const Color(
      0xFF1C1C1E,
    ); // dunkles Grau/Schwarz
    final Color cardColor = const Color(0xFF2C2C2E); // etwas helleres Grau
    final Color accentColor = const Color(0xFF7F5AF0); // lila / highlight
    final Color textColor = Colors.white; // für helle Schrift

    return Scaffold(
      // Wir verzichten mal auf die AppBar und machen ein custom-Design
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
                  // Optional: Avatar oder Icon
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

              // Beispiel für Statistiken oder "Karten"
              // (könntest du später mit Graphen / Stats befüllen)
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
                      'Bisher 0 Boulder analysiert',
                      style: TextStyle(
                        color: textColor.withOpacity(0.7),
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 40),

              // Buttons
              // => Anstelle normaler ElevatedButton => custom styles
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
