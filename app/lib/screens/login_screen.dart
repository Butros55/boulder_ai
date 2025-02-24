import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  LoginScreenState createState() => LoginScreenState();
}

class LoginScreenState extends State<LoginScreen> {
  final Color backgroundColor = const Color(0xFF1C1C1E);
  final Color cardColor = const Color(0xFF2C2C2E);
  final Color accentColor = const Color(0xFF7F5AF0);
  final Color textColor = Colors.white;

  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  final storage = const FlutterSecureStorage();
  bool _isLoading = false;
  String? _errorMessage;

  Future<void> _login() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });
    final username = _usernameController.text.trim();
    final password = _passwordController.text;
    try {
      final response = await http.post(
        Uri.parse("http://127.0.0.1:5000/login"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"username": username, "password": password}),
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final token = data["access_token"];
        await storage.write(key: 'jwt_token', value: token);
        Navigator.pushReplacementNamed(context, '/');
      } else {
        final data = jsonDecode(response.body);
        setState(() {
          _errorMessage = data["msg"] ?? "Login fehlgeschlagen";
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Fehler: $e";
      });
    }
    setState(() {
      _isLoading = false;
    });
  }

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: backgroundColor,
      body: SafeArea(
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
                  CircleAvatar(
                    backgroundColor: cardColor,
                    child: Icon(Icons.person, color: textColor),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              Text(
                'Willkommen zurück!\nBitte logge dich ein.',
                style: TextStyle(
                  color: textColor.withValues(alpha: 0.8),
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 40),
              if (_errorMessage != null)
                Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.redAccent),
                ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: cardColor,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: TextField(
                  controller: _usernameController,
                  style: TextStyle(color: textColor),
                  decoration: InputDecoration(
                    hintText: 'Benutzername',
                    hintStyle: TextStyle(color: textColor.withValues(alpha: 0.7)),
                    border: InputBorder.none,
                  ),
                ),
              ),
              
              const SizedBox(height: 16),

              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: cardColor,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: TextField(
                  controller: _passwordController,
                  obscureText: true,
                  style: TextStyle(color: textColor),
                  decoration: InputDecoration(
                    hintText: 'Passwort',
                    hintStyle: TextStyle(color: textColor.withValues(alpha: 0.7)),
                    border: InputBorder.none,
                  ),
                ),
              ),
              const SizedBox(height: 24),

              GestureDetector(
                onTap: _isLoading ? null : _login,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [accentColor.withValues(alpha: 0.8), accentColor],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Center(
                    child:
                        _isLoading
                            ? CircularProgressIndicator(
                              valueColor: AlwaysStoppedAnimation<Color>(
                                textColor,
                              ),
                            )
                            : Text(
                              'Login',
                              style: TextStyle(
                                color: textColor,
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                  ),
                ),
              ),
              const SizedBox(height: 16),

              Center(
                child: GestureDetector(
                  onTap: () {
                    Navigator.pushReplacementNamed(context, '/register');
                  },
                  child: Text(
                    'Noch keinen Account? Registriere dich',
                    style: TextStyle(
                      color: accentColor,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
