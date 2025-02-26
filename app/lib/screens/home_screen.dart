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
  final storage = const FlutterSecureStorage();

  bool _isLoading = false;
  int _totalAnalyses = 0;
  List<dynamic> _recentAnalyses = [];

  @override
  void initState() {
    super.initState();
    _fetchTotalAnalyses();
  }

  Future<void> _logout() async {
    await storage.delete(key: 'jwt_token');
    Navigator.pushReplacementNamed(context, '/login');
  }

  Future<String?> getToken() async {
    return await storage.read(key: 'jwt_token');
  }

  void _showLogoutError() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text(
          "Deine Sitzung ist abgelaufen. Bitte logge dich erneut ein.",
          style: TextStyle(color: Colors.white),
        ),
        backgroundColor: Colors.red,
        duration: Duration(seconds: 3),
      ),
    );
  }

  Future<void> _fetchTotalAnalyses() async {
    try {
      final token = await getToken();
      if (token == null) {
        debugPrint(
          "Kein Token gefunden, Analysen können nicht geladen werden.",
        );
        _showLogoutError();
        _logout();
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
          _recentAnalyses = data['analyses'] as List<dynamic>;
        });
      } else if (response.statusCode == 401) {
        debugPrint("Token ungültig oder abgelaufen.");
        _showLogoutError();
        _logout();
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
        setState(() => _isLoading = true);
        dynamic imageData;
        if (kIsWeb) {
          imageData = await picture.readAsBytes();
        } else {
          imageData = File(picture.path);
        }
        final result = await ImageProcessor.processImage(imageData);
        setState(() => _isLoading = false);
        Navigator.pushNamed(
          context,
          '/result',
          arguments: result,
        ).then((_) => _fetchTotalAnalyses());
      }
    } catch (e) {
      debugPrint('Fehler beim Aufnehmen des Fotos: $e');
      setState(() => _isLoading = false);
    }
  }

  Future<void> _pickImage() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
      );
      if (pickedFile != null) {
        setState(() => _isLoading = true);
        dynamic imageData;
        if (kIsWeb) {
          imageData = await pickedFile.readAsBytes();
        } else {
          imageData = File(pickedFile.path);
        }
        final result = await ImageProcessor.processImage(imageData);
        setState(() => _isLoading = false);
        Navigator.pushNamed(
          context,
          '/result',
          arguments: result,
        ).then((_) => _fetchTotalAnalyses());
      }
    } catch (e) {
      debugPrint('Fehler beim Auswählen eines Bildes: $e');
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final Color backgroundColor = const Color(0xFF1C1C1E);
    final Color cardColor = const Color(0xFF2C2C2E);
    final Color accentColor = const Color(0xFF7F5AF0);
    final Color textColor = Colors.white;

    return Scaffold(
      backgroundColor: backgroundColor,
      body: Stack(
        children: [
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
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
                      PopupMenuButton<String>(
                        icon: CircleAvatar(
                          backgroundColor: cardColor,
                          child: Icon(Icons.person, color: textColor),
                        ),
                        onSelected: (value) {
                          if (value == 'logout') {
                            _logout();
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: const Text(
                                  "Sie wurden erfolgreich abgemeldet!",
                                  style: TextStyle(color: Colors.white),
                                ),
                                backgroundColor: Colors.green,
                                duration: const Duration(seconds: 3),
                              ),
                            );
                          }
                        },
                        color: cardColor,
                        itemBuilder:
                            (context) => [
                              PopupMenuItem(
                                value: 'logout',
                                child: Text(
                                  'Logout',
                                  style: TextStyle(color: textColor),
                                ),
                              ),
                            ],
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),

                  Text(
                    'Willkommen! \nErkenne Boulder-Griffe per Foto oder Bild-Upload.',
                    style: TextStyle(
                      color: textColor.withValues(alpha: 0.8),
                      fontSize: 16,
                    ),
                  ),

                  const SizedBox(height: 32),

                  Card(
                    color: cardColor,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: 2,
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Row(
                        children: [
                          Icon(Icons.insights, color: accentColor, size: 36),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
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
                                    color: textColor.withValues(alpha: 0.7),
                                    fontSize: 14,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 32),

                  Card(
                    color: cardColor,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: 2,
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        children: [
                          Text(
                            'Aktionen',
                            style: TextStyle(
                              color: textColor,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Row(
                            children: [
                              Expanded(
                                child: _buildOptionButton(
                                  label: 'Foto aufnehmen',
                                  icon: Icons.camera_alt_rounded,
                                  onTap: _takePicture,
                                  accentColor: accentColor,
                                  textColor: textColor,
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: _buildOptionButton(
                                  label: 'Bild hochladen',
                                  icon: Icons.photo_library_rounded,
                                  onTap: _pickImage,
                                  accentColor: accentColor,
                                  textColor: textColor,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 32),

                  Text(
                    'Zuletzt analysierte Bilder',
                    style: TextStyle(
                      color: textColor,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const SizedBox(height: 16),

                  _buildRecentAnalysesList(cardColor, textColor),
                ],
              ),
            ),
          ),
          if (_isLoading)
            ModalBarrier(
              dismissible: false,
              color: Colors.black.withValues(alpha: 0.5),
            ),
          if (_isLoading)
            Center(
              child: CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(accentColor),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildRecentAnalysesList(Color cardColor, Color textColor) {
    if (_recentAnalyses.isEmpty) {
      return Text(
        'Noch keine Analysen vorhanden.',
        style: TextStyle(color: textColor.withValues(alpha: 0.7)),
      );
    }
    final Color accentColor = const Color(0xFF7F5AF0);

    return Wrap(
      crossAxisAlignment: WrapCrossAlignment.center,
      alignment: WrapAlignment.center,
      runAlignment: WrapAlignment.center,
      spacing: 16,
      runSpacing: 16,
      children:
          _recentAnalyses.map((analysis) {
            final String originalBase64 = analysis['original_image'];
            final String timestamp = analysis['timestamp'];
            final List<dynamic> detections = analysis['detections'];
            final int detectionsTotal = detections.length;
            final bytes = base64Decode(originalBase64);

            return SizedBox(
              width: 260,
              child: Card(
                color: cardColor,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                elevation: 2,
                child: Column(
                  children: [
                    ClipRRect(
                      borderRadius: const BorderRadius.only(
                        topLeft: Radius.circular(12),
                        topRight: Radius.circular(12),
                      ),
                      child: Image.memory(
                        bytes,
                        width: 250,
                        height: 350,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(Icons.assessment, color: accentColor),
                                  const SizedBox(width: 8),
                                  Text(
                                    '${analysis['id']}',
                                    style: TextStyle(
                                      color: textColor,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 4),
                              Row(
                                children: [
                                  Icon(Icons.access_time, color: accentColor),
                                  const SizedBox(width: 8),
                                  Text(
                                    'Zeit: $timestamp',
                                    style: TextStyle(
                                      color: textColor.withValues(alpha: 0.7),
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 4),
                              Row(
                                children: [
                                  Icon(
                                    Icons.fitness_center,
                                    color: accentColor,
                                  ),
                                  const SizedBox(width: 8),
                                  Text(
                                    'Griffe: $detectionsTotal',
                                    style: TextStyle(
                                      color: textColor.withValues(alpha: 0.7),
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    Align(
                      alignment: Alignment.centerRight,
                      child: IconButton(
                        icon: Icon(Icons.arrow_forward, color: textColor),
                        onPressed: () {
                          Navigator.pushNamed(
                            context,
                            '/result',
                            arguments: {
                              "original_image": analysis['original_image'],
                              "detections": analysis['detections'],
                              "routes": analysis['routes'],
                              "image_width": analysis['image_width'] ?? 800,
                              "image_height": analysis['image_height'] ?? 600,
                            },
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
    );
  }

  Widget _buildOptionButton({
    required String label,
    required IconData icon,
    required VoidCallback onTap,
    required Color accentColor,
    required Color textColor,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [accentColor.withValues(alpha: 0.8), accentColor],
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
    );
  }
}
