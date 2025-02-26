import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  RegisterScreenState createState() => RegisterScreenState();
}

class RegisterScreenState extends State<RegisterScreen> {
  final Color backgroundColor = const Color(0xFF1C1C1E);
  final Color cardColor = const Color(0xFF2C2C2E);
  final Color accentColor = const Color(0xFF7F5AF0);
  final Color textColor = Colors.white;

  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();

  // FocusNodes für die Textfelder
  final FocusNode _usernameFocus = FocusNode();
  final FocusNode _passwordFocus = FocusNode();
  final FocusNode _confirmPasswordFocus = FocusNode();

  bool _isLoading = false;
  String? _errorMessage;
  String? _successMessage;

  Future<void> _register() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _successMessage = null;
    });
    final username = _usernameController.text.trim();
    final password = _passwordController.text;
    final confirmPassword = _confirmPasswordController.text;
    if (password != confirmPassword) {
      setState(() {
        _errorMessage = "Passwörter stimmen nicht überein.";
        _isLoading = false;
      });
      return;
    }
    try {
      final response = await http.post(
        Uri.parse("http://127.0.0.1:5000/register"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"username": username, "password": password}),
      );
      if (response.statusCode == 201) {
        Navigator.pushReplacementNamed(context, '/login');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text(
              "Sie haben sich erfolgreich registriert, bitte melden Sie sich an!",
              style: TextStyle(color: Colors.white),
            ),
            backgroundColor: Colors.green,
            duration: const Duration(seconds: 3),
          ),
        );
      } else {
        final data = jsonDecode(response.body);
        setState(() {
          _errorMessage = data["msg"] ?? "Registrierung fehlgeschlagen";
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
    _confirmPasswordController.dispose();
    _usernameFocus.dispose();
    _passwordFocus.dispose();
    _confirmPasswordFocus.dispose();
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
                'Registriere dich und starte durch!',
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
              if (_successMessage != null)
                Text(_successMessage!, style: TextStyle(color: accentColor)),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: cardColor,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: TextField(
                  controller: _usernameController,
                  focusNode: _usernameFocus,
                  style: TextStyle(color: textColor),
                  decoration: InputDecoration(
                    hintText: 'Benutzername',
                    hintStyle: TextStyle(color: textColor.withValues(alpha: 0.7)),
                    border: InputBorder.none,
                  ),
                  onSubmitted: (_) {
                    FocusScope.of(context).requestFocus(_passwordFocus);
                  },
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
                  focusNode: _passwordFocus,
                  obscureText: true,
                  style: TextStyle(color: textColor),
                  decoration: InputDecoration(
                    hintText: 'Passwort',
                    hintStyle: TextStyle(color: textColor.withValues(alpha: 0.7)),
                    border: InputBorder.none,
                  ),
                  onSubmitted: (_) {
                    FocusScope.of(context).requestFocus(_confirmPasswordFocus);
                  },
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
                  controller: _confirmPasswordController,
                  focusNode: _confirmPasswordFocus,
                  obscureText: true,
                  style: TextStyle(color: textColor),
                  decoration: InputDecoration(
                    hintText: 'Passwort bestätigen',
                    hintStyle: TextStyle(color: textColor.withValues(alpha: 0.7)),
                    border: InputBorder.none,
                  ),
                  onSubmitted: (_) {
                    _register();
                  },
                ),
              ),
              const SizedBox(height: 24),
              GestureDetector(
                onTap: _isLoading ? null : _register,
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
                    child: _isLoading
                        ? CircularProgressIndicator(
                            valueColor: AlwaysStoppedAnimation<Color>(
                              textColor,
                            ),
                          )
                        : Text(
                            'Registrieren',
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
                    Navigator.pushReplacementNamed(context, '/login');
                  },
                  child: Text(
                    'Bereits einen Account? Zum Login',
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
