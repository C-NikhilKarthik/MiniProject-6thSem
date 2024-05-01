import React, { useState } from 'react';
import { View, TextInput, Text, Button, StyleSheet, TouchableOpacity } from 'react-native';
import axios from 'axios';

const RegistrationPage = ({ navigation }) => {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleRegistration = async () => {
        try {
            const response = await axios.post('http://10.0.3.61:5000/register', {
                "name": name,
                "email": email,
                "password": password
            });

            if (response.data.status) {
                navigation.navigate('loginPage');
            } else {
                setError(response.data.message);
            }
        } catch (error) {
            console.error('Error registering:', error);
            setError('An error occurred. Please try again later.');
        }
    };

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Login</Text>
            <TextInput
                style={styles.input}
                placeholder="Email"
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
            />
            <TextInput
                style={styles.input}
                placeholder="Name"
                value={name}
                onChangeText={setName}
            />
            <TextInput
                style={styles.input}
                placeholder="Password"
                value={password}
                onChangeText={setPassword}
                secureTextEntry
            />
            {error ? <Text style={styles.error}>{error}</Text> : null}
            <TouchableOpacity style={styles.button} onPress={handleRegistration}>
                <Text style={styles.buttonText}>Register</Text>
            </TouchableOpacity>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: 20,
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 20,
    },
    input: {
        width: '100%',
        height: 40,
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 5,
        paddingHorizontal: 10,
        marginBottom: 20,
    },
    button: {
        width: '100%',
        height: 40,
        backgroundColor: 'blue',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 5,
        marginBottom: 10,
    },
    buttonText: {
        color: 'white',
        fontWeight: 'bold',
    },
    registerButton: {
        width: '100%',
        height: 40,
        backgroundColor: 'green',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 5,
    },
    registerButtonText: {
        color: 'white',
        fontWeight: 'bold',
    },
    error: {
        color: 'red',
        marginBottom: 10,
    },
});

export default RegistrationPage;
